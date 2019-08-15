"""functions and classes for inference."""

from collections import OrderedDict
from typing import Tuple

import torch
from torch.nn import Module
import numpy as np
from PIL import Image
from ml_utils.prediction_filtering import (
    PredictionFilterPipeline,
    ConfidenceFilter,
    MaxDetFilter,
    NMSFilter,
)

from .data.encoding import frcnn_box_decode
from .utils import make_input_transform, tensor_to_ndarray


class Detector:
    """perform "detect" part of detect-to-track inference."""

    def __init__(
        self,
        model: Module,
        anchors: np.ndarray,
        input_dims: Tuple[int, int],
        rpn_conf_thresh: float,
        rpn_max_dets: int,
        rpn_nms_thresh: float,
        rcnn_conf_thresh: float,
    ) -> None:
        self.model = model.cuda()
        self.anchors = anchors
        self._im_to_x = make_input_transform(input_dims)
        self.region_filter = PredictionFilterPipeline(
            ConfidenceFilter(rpn_conf_thresh),
            MaxDetFilter(rpn_max_dets),
            NMSFilter(rpn_nms_thresh),
        )
        self.rcnn_conf_thresh = rcnn_conf_thresh

    def _filter_rcnn_output(
        self, confs: np.ndarray, bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """filter RCNN detections based on non-background confidence scores."""
        conf_mask = confs[:, 1:].sum(1) > self.rcnn_conf_thresh
        confs = confs[conf_mask, :]
        bboxes = bboxes[conf_mask, :]

        return confs, bboxes

    def __call__(
        self, im0: Image.Image, im1: Image.Image
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            im0: first frame.
            im1: second frame.

        Returns:
            confs0: (|D0|, |C|+1); classwise confidences for first frame.
            confs1: (|D1|, |C|+1); classwise confidences for second frame.
            bboxes0: (|D0|, 4); bounding boxes for first frame.
            bboxes1: (|D1|, 4); bounding boxes for second frame.
            tracks: (|D0|, 4); cross-frame tracks.
        """
        # prepare backbone inputs.
        x0, x1 = (self._im_to_x(im) for im in (im0, im1))
        x = torch.stack((x0, x1)).cuda()

        # backbone: images -> feature map pyramids.
        fmaps = self.model.backbone(x)

        # RPN: C4 -> (confidences, anchor offsets, FPN FM).
        o, b_rpn, fm_reg = self.model.rpn(fmaps["c4"])
        o0, o1 = (tensor_to_ndarray(confs) for confs in o[:, :, 1])
        rboxes0, rboxes1 = (
            frcnn_box_decode(self.anchors, tensor_to_ndarray(offsets))
            for offsets in b_rpn
        )
        _, rboxes0 = self.region_filter(o0, rboxes0)
        _, rboxes1 = self.region_filter(o1, rboxes1)

        # prepare RCNN inputs.
        fm5_0, fm5_1 = fmaps["c5"]
        rboxes0, rboxes1 = (
            torch.as_tensor(rboxes, dtype=torch.float32).cuda()
            for rboxes in (rboxes0, rboxes1)
        )

        # RCNN: (C5, region proposals) -> (class confidences, region offsets).
        c0, b0_rcnn = self.model.rcnn(fm5_0, rboxes0)
        c1, b1_rcnn = self.model.rcnn(fm5_1, rboxes1)

        # process RCNN outputs.
        confs0, confs1 = (tensor_to_ndarray(c_hat) for c_hat in (c0, c1))
        bboxes0 = frcnn_box_decode(
            tensor_to_ndarray(rboxes0), tensor_to_ndarray(b0_rcnn)
        )
        bboxes1 = frcnn_box_decode(
            tensor_to_ndarray(rboxes1), tensor_to_ndarray(b1_rcnn)
        )
        confs0, bboxes0 = self._filter_rcnn_output(confs0, bboxes0)
        confs1, bboxes1 = self._filter_rcnn_output(confs1, bboxes1)

        # prepare correlation tracker inputs.
        (fm3_0, fm3_1), (fm4_0, c4_1), (fm5_0, fm5_1) = (
            fmaps[f"c{n}"] for n in range(3, 6)
        )
        fms0 = OrderedDict(c3=fm3_0, c4=fm4_0, c5=fm5_0)
        fms1 = OrderedDict(c3=fm3_1, c4=c4_1, c5=fm5_1)
        fm_reg0, fm_reg1 = fm_reg
        track_rois = torch.as_tensor(bboxes0, dtype=torch.float32).cuda()

        # corr. tracker: (fm_pyramids, RPN FMs, bounding boxes) -> cross-frame tracks.
        t = self.model.c_tracker(fms0, fms1, fm_reg0, fm_reg1, track_rois)
        tracks = tensor_to_ndarray(t)

        return confs0, confs1, bboxes0, bboxes1, tracks
