"""encoder objects for converting human-readable labels to network targets"""

import abc
from typing import Tuple, Sequence

import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
from ml_utils.bbox_utils import ijhw_to_ijij, compute_ious

from . import ObjectLabel


class FRCNNLabelEncoder(abc.ABC):
    """handles encoding of ground-truth labels according to
    https://arxiv.org/abs/1506.01497"""
    @abc.abstractmethod
    def __call__(self, labels):
        raise NotImplementedError


class AnchorEncoder(FRCNNLabelEncoder):
    """assign target classes and bounding box offsets anchorwise based on
    iou with closest ground-truth bounding box.

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
        self.anchors = anchors
        self._iou_thresh = iou_thresh
        self._iou_margin = iou_margin

        ### determine which anchors cross image boundaries. These are ignored
        ### during training.
        anchors_ijij = ijhw_to_ijij(self.anchors)
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

        Args:
            labels: ground-truth object label for each object in image.

        Returns:
            loss_weights: (|A|,) anchorwise weight assigned to losses for
                training.
            c_star: (|A|,) anchorwise assigned class values.
            b_star: (|A|, 4) anchorwise offset from assigned bounding box.
        """
        ### unzip labels
        boxes = np.array([label.box for label in labels])  # (|O|, 4)

        ### assign ground-truth boxes anchorwise
        ious = compute_ious(self.anchors, boxes)  # (|A|, |B|)
        anchwise_best_gt_ind = ious.argmax(1)  # (|A|,)
        anchwise_best_iou = ious.max(1)  # (|A|,)

        ### loss_weights encoding
        loss_weights = np.logical_and(
            np.abs(anchwise_best_iou - self._iou_thresh) > self._iou_margin,
            np.logical_not(self._crosses_boundary)
        )  # (|A|,)

        ### c_star encoding
        is_best_anchor = np.zeros(len(self.anchors))  # (|A|,)
        is_best_anchor[ious.argmax(0)] = 1
        c_star = np.logical_or(
            anchwise_best_iou > self._iou_thresh, is_best_anchor
        )

        ### b_star encoding
        b_star = frcnn_box_encode(
            self.anchors, boxes[anchwise_best_gt_ind, :]
        )  # (|A|, 4)

        return loss_weights, c_star, b_star


class RegionEncoder(FRCNNLabelEncoder):
    """assign target classes and bounding box offsets regionwise based on
    iou with closest ground-truth bounding box.

    Args:
        iou_thresh: if iou(region, gt-bbox) > iou_thresh for any gt-bbox,
            region is assigned a class.
    """
    def __init__(self, iou_thresh: float) -> None:
        self._iou_thresh = iou_thresh

    def __call__(
            self,
            regions: np.ndarray,
            labels: Sequence[ObjectLabel]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """encode human-readable ground-truth frame instances into
        network-readable frame instances, according to Faster RCNN conventions.

        Args:
            regions: (|R|, 4) predicted region proposal boxes.
            labels: ground-truth object label for each object in image.

        Returns:
            c_star: (|R|,) regionwise class assignments.
            b_star: (|R|, 4) regionwise bounding box offsets from assigned
                ground-truth box.
        """
        ### unzip labels
        classes = np.array([label.class_id for label in labels])
        boxes = np.array([label.bbox for label in labels])

        ### assign ground-truth boxes regionwise
        ious = compute_ious(regions, boxes)  # (|A|, |B|)
        regionwise_best_gt = ious.argmax(1)  # (|A|,)
        regionwise_best_iou = ious.max(1)  # (|A|,)

        ### c_star encoding
        c_star = classes[regionwise_best_gt]  # (|A|,)
        c_star[regionwise_best_iou < self._iou_thresh] = 0

        ### b_star encoding
        b_star = frcnn_box_encode(
            regions, boxes[regionwise_best_gt, :]
        )  # (|A|, 4)

        return c_star, b_star


def make_input_transform(net_input_hw: int):
    """return transform layer for PIL Image -> network input"""
    return Compose([
        Resize((net_input_hw, net_input_hw)),
        ToTensor
    ])


def frcnn_box_encode(anchors: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """given boxes and anchors, return bounding box offsets from anchors.
    equations for encoding anchor offsets:
        ti = (bi - ai) / ah
        tj = (bj - aj) / aw
        th = log(bh / ah)
        tw = log(bw / aw)

    Args:
        anchors: (|A|, 4) bounding box priors. can be deterministically
            precomputed anchors or predicted rois.
        boxes: (|A|, 4) anchorwise closest ground truth bounding box.

    Returns:
        offsets: (|A|, 4) anchor offsets from closest bounding box.
    """
    b_ij, b_hw = np.hsplit(boxes, 2)  # 2*(|A|, 2)
    a_ij, a_hw = np.hsplit(anchors, 2)  # 2*(|A|, 2)
    t_ij = (b_ij - a_ij) / a_hw   # (|A|, 2)
    t_hw = np.log(b_hw / a_hw)  # (|A|, 2)
    offsets = np.concatenate([t_ij, t_hw], axis=1)  # (|A|, 4)

    return offsets


def frcnn_box_decode(anchors: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """given anchors and anchor offsets, return bounding box coordinates
    equations for decoding anchor offsets:
        bi = ti * ah + ai
        bj = tj * aw + aj
        th = exp(th) * ah
        tw = exp(tw) * aw

    Args:
        anchors: (|A|, 4) bounding box priors. can be deterministically
            precomputed anchors or predicted rois.
        offsets: (|A|, 4) predicted offsets from anchors.

    Returns:
        boxes: (|A|, 4) bounding box coordinates.
    """
    t_ij, t_hw = np.hsplit(offsets, 2)  # 2*(|A|, 2)
    a_ij, a_hw = np.hsplit(anchors, 2)  # 2*(|A|, 2)

    b_ij = t_ij * a_hw + a_ij  # (|A|, 2)
    b_hw = np.exp(t_hw) * a_hw  # (|A|, 2)

    boxes = np.concatenate([b_ij, b_hw], axis=1)  # (|A|, 4)

    return boxes


class TrackEncoder:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError
