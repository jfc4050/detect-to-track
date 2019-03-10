"""encoder objects for converting human-readable labels to network targets"""

from typing import Tuple, Sequence

import numpy as np
from ml_utils.bbox_utils import ijhw_to_ijij, compute_ious

from . import ObjectLabel


class FRCNNEncoder:
    """anchorwise class and bounding-box offset encoding
    according to https://arxiv.org/abs/1506.01497

    Args:
        anchors: (|A|, 4); bounding box priors. must be ijhw and fractional.
        iou_thresh: if iou(anchor, gt-bbox) > iou_thresh for any gt-bbox,
            anchor is marked positive.
        iou_margin: if |iou(anchor, best-gt-bbox) - iou_thresh| < iou_margin,
            losses for this anchor are ignored during training.
    """
    def __init__(
            self,
            anchors: np.ndarray,
            iou_thresh: float = 0.5,
            iou_margin: float = 0.2
    ):
        self._anchors = anchors
        self._iou_thresh = iou_thresh
        self._iou_margin = iou_margin

        ### determine which anchors cross image boundaries. These are ignored
        ### during training.
        anchors_ijij = ijhw_to_ijij(self._anchors)
        self._crosses_boundary = np.logical_or(
            np.any(anchors_ijij <= 0, axis=1),
            np.any(anchors_ijij >= 1, axis=1)
        )

    def __call__(
            self,
            labels: Sequence[ObjectLabel]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """encode human-readable ground-truth frame instances into
        network-readable frame instances, according to Faster RCNN conventions.

        satisfy one of two conditions -> anchor a marked positive:
            i) IoU(a, b) > iou_thresh for any ground-truth box b
            ii) a is the anchor with the highest IoU
                for some ground-truth box b

        equations for encoding anchor offsets:
            ti = (bi - ai) / ah
            tj = (bj - aj) / aw
            th = log(bh / ah)
            tw = log(bw / aw)

        Args:
            labels: sequence of object labels.

        Returns:
            loss_weights: (|A|,) anchorwise weight assigned to losses for
                training.
            c_star: (|A|,) anchorwise assigned class values.
            b_star: (|A|, 4) anchorwise offset from assigned bounding box.
        """
        ### start with label conversion
        # classes: (|O|,), bboxes: (|O|, 4)
        classes = np.array([label.class_id for label in labels])
        bboxes = np.array([label.bbox for label in labels])

        ious = compute_ious(self._anchors, bboxes)  # (|A|, |B|)
        anchwise_best_gt_ind = ious.argmax(1)  # (|A|,)
        anchwise_best_iou = ious.max(1)  # (|A|,)

        ### loss_weights encoding
        loss_weights = np.logical_and(
            np.abs(anchwise_best_iou - self._iou_thresh) > self._iou_margin,
            np.logical_not(self._crosses_boundary)
        )  # (|A|,)

        ### c_star encoding
        c_star = classes[anchwise_best_gt_ind]  # (|A|,)
        is_best_anchor = np.zeros(len(self._anchors))  # (|A|,)
        is_best_anchor[ious.argmax(0)] = 1
        neg_mask = np.logical_and(
            anchwise_best_iou < self._iou_thresh, np.logical_not(is_best_anchor)
        )  # (|A|,)
        c_star[neg_mask] = 0

        ### b_star encoding
        b_ij, b_hw = np.hsplit(bboxes[anchwise_best_gt_ind, :], 2)  # 2*(|A|, 2)
        a_ij, a_hw = np.hsplit(self._anchors, 2)  # 2*(|A|, 2)
        t_ij = (b_ij - a_ij) / a_hw   # (|A|, 2)
        t_hw = np.log(b_hw / a_hw)  # (|A|, 2)
        b_star = np.concatenate([t_ij, t_hw], axis=1)  # (|A|, 4)

        return loss_weights, c_star, b_star


class TrackEncoder:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError
