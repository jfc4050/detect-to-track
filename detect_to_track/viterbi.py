"""detection linking utilities."""

from copy import deepcopy
from typing import Sequence, List, Tuple

import numpy as np
from ml_utils.boundingboxes import compute_ious
from ml_utils.sequence import sliding_window


def compute_link_scores(
    confs_a: np.ndarray,
    confs_b: np.ndarray,
    bboxes_a: np.ndarray,
    bboxes_b: np.ndarray,
    tracks: np.ndarray,
    iou_thresh: float,
) -> np.ndarray:
    """computes linking scores for all potential links in A x B.
    s(d1, d2, t) = c(d1) + c(d2) + psi(b(d1), b(d2), t)
    where psi(b(d1), b(d2), t) == IoU(b(d1), b(d2), t) > iou_thresh

    Args:
        confs_a: (|A|,); detection confidences in frame A.
        confs_b: (|B|,); detection confidences in frame B.
        bboxes_a: (|A|, 4); bounding boxes in frame A.
        bboxes_b: (|B|, 4); bounding boxes in frame B.
        tracks: (|T|, 4); predicted tracks between frames A and B.

    Returns:
        link_scores: (|A|, |B|); link scores for all potential links in A x B.
    """
    confs = confs_a[:, None] + confs_b  # (|A|, |B|)

    matches_a = compute_ious(bboxes_a, tracks) > iou_thresh  # (|A|, |T|)
    matches_b = compute_ious(bboxes_b, tracks) > iou_thresh  # (|B|, |T|)
    matches = np.logical_and(matches_a[:, None, :], matches_b)  # (|A|, |B|, |T|)
    psi = np.any(matches, axis=-1).astype(float)  # (|A|, |B|)

    scores = confs + psi

    return scores


def compute_score_seq(
    conf_seq: Sequence[np.ndarray],
    bbox_seq: Sequence[np.ndarray],
    track_seq: Sequence[np.ndarray],
    iou_thresh: float,
) -> List[np.ndarray]:
    """compute sequence of score matrices."""
    if len(conf_seq) != len(bbox_seq):
        raise ValueError(
            f"recieved |conf_seq|={len(conf_seq)}, but |bbox_seq|={len(bbox_seq)}"
        )
    if len(track_seq) != len(conf_seq) - 1:
        raise ValueError(
            f"recieved |track_seq|={len(track_seq)} but |det_seq|={len(conf_seq)}"
        )

    T = [
        compute_link_scores(c1, c2, b1, b2, t, iou_thresh)
        for ((c1, b1), (c2, b2)), t in zip(
            sliding_window(zip(conf_seq, bbox_seq), 2), track_seq
        )
    ]

    return T


def viterbi(
    score_seq: List[np.ndarray], init_scores: Sequence[float] = None
) -> Tuple[List[int], int]:
    """viterbi algorithm implemented using top-down dynamic programming.

    Args:
        score_seq: sequence of transition matrices, each having shape (|D1|, |D2|).
            these contain the transition scores between each possible linking
            of detections in adjacent time steps.
        init_scores: scores for initial states.

    Returns:
        path: optimal path to final time step.
        score: score of optimal path.
    """

    if not score_seq and init_scores is None:
        raise ValueError(f"if no transitions init_scores need to be passed in.")

    n_time_steps = len(score_seq) + 1

    init_scores = init_scores or [0.0] * score_seq[0].shape[0]

    ans = [([src_node], init_score) for src_node, init_score in enumerate(init_scores)]
    for ts in range(1, n_time_steps):
        transitions = score_seq[ts - 1]
        _, n_dst_nodes = transitions.shape

        ans_ts = list()
        for dst_node in range(n_dst_nodes):
            best_score, best_path = 0.0, [dst_node]
            for src_node, trans_score in enumerate(transitions[:, dst_node]):
                src_path, src_score = ans[src_node]
                score = src_score + trans_score

                if score > best_score:
                    best_score = score
                    best_path = src_path + [dst_node]

            ans_ts.append((best_path, best_score))

        ans = ans_ts

    path, score = max(ans, key=lambda x: x[1])

    return path, score


def viterbi_multi_link(
    score_seq: List[np.ndarray], init_scores: List[float] = None
) -> List[Tuple[Tuple[int, int], float, List[int]]]:
    """generate multiple paths through sequence of states.
    loop:
        1) find best path using viterbi algorithm.
        2) remove nodes that were part of best path.
    """
    score_seq, init_scores = deepcopy(score_seq), deepcopy(init_scores)
    if not score_seq and init_scores is None:
        raise ValueError(f"if no transitions, init_scores need to be passed in.")
    init_scores = init_scores or [0.0] * len(score_seq[0])

    n_time_steps = len(score_seq) + 1

    ans = list()
    for final_ts in reversed(range(1, n_time_steps)):
        # get all tubelets terminating at timestep final_ts.
        while np.any(np.isfinite(score_seq[final_ts - 1])):
            # find best path using viterbi algorithm.
            track_path, track_score = viterbi(score_seq, init_scores)
            start_ts = final_ts - len(track_path) + 1
            ans.append(((start_ts, final_ts), track_score, track_path))

            # "remove" nodes that were part of best path.
            for ts, ts_node in zip(range(start_ts, final_ts + 1), track_path):
                if ts == 0:
                    init_scores[ts_node] = -np.inf
                if ts > 0:
                    score_seq[ts - 1][:, ts_node] = -np.inf  # incoming transitions.
                if ts < final_ts:
                    score_seq[ts][ts_node, :] = -np.inf  # outgoing transitions.

        score_seq.pop()

    # handle tubelets terminating at timestep 0.
    for node, node_score in enumerate(init_scores):
        if np.isfinite(node_score):
            ans.append(((0, 0), node_score, [node]))

    return ans


def viterbi_tracking(
    conf_seq: List[np.ndarray],
    bbox_seq: List[np.ndarray],
    track_seq: List[np.ndarray],
    iou_thresh: float,
    min_len: int,
) -> List[Tuple[Tuple[int, int], np.ndarray]]:
    init_scores = conf_seq[0].tolist()
    T = compute_score_seq(conf_seq, bbox_seq, track_seq, iou_thresh)

    track_paths = viterbi_multi_link(T, init_scores)

    tubelets = list()
    for (start_ts, end_ts), track_score, track_path in track_paths:
        if end_ts - start_ts + 1 >= min_len:
            tubelet = np.array(
                [
                    bbox_seq[ts][ts_node]
                    for ts, ts_node in zip(range(start_ts, end_ts + 1), track_path)
                ]
            )
            tubelets.append(((start_ts, end_ts), tubelet))

    return tubelets
