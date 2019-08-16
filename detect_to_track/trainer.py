"""handles joint training of entire system"""

import math
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, Sequence

import torch
from torch.utils.data import BatchSampler, RandomSampler
from torch.optim import SGD
import numpy as np
import wandb
from PIL import Image
from ml_utils.prediction_filtering import (
    PredictionFilterPipeline,
    ConfidenceFilter,
    MaxDetFilter,
    NMSFilter,
)
from ml_utils.vis_utils import draw_detections

from .data.types import DataManager, ImageInstance
from .data.encoding import AnchorEncoder, RegionEncoder, frcnn_box_decode, track_encode
from .loss import RPNLoss, RCNNLoss, TrackLoss
from .models import DetectTrackModule
from .inference import Detector
from .utils import DTLoss, build_anchors, tensor_to_ndarray, make_input_transform


class BatchLoader:
    """was easier to just write this class instead of adapting a torch DataLoader."""

    def __init__(self, data_manager: DataManager, batch_size: int) -> None:
        self.data_manager = data_manager
        self.idx_sampler = BatchSampler(
            RandomSampler(data_manager), batch_size, drop_last=True
        )

    def __iter__(self) -> object:
        for indices in self.idx_sampler:
            batch = [self.data_manager[idx] for idx in indices]
            yield batch


class DetectTrackTrainer:
    """approximate joint training for two stage object detectors.
    ignores detector head loss wrt to region proposal.
    this can be (but is not currently) addressed by substituting the
    ROIPooling layer for a ROIWarping layer.
    see https://arxiv.org/abs/1506.01497.
    """

    def __init__(
        self,
        model: DetectTrackModule,
        trn_manager: DataManager,
        val_manager: DataManager,
        rep_manager: DataManager,
        batch_size: int,
        input_dims: Tuple[int, int],
        fm_stride: int,
        anchor_areas: Sequence[float],
        anchor_aspect_ratios: Sequence[float],
        encoder_iou_thresh: float,
        encoder_iou_margin: float,
        trn_roi_conf_thresh: float,
        trn_roi_max_dets: int,
        trn_roi_nms_iou_thresh: float,
        alpha: float,
        gamma: float,
        loss_coefs: Sequence[float],
        sgd_kwargs: dict,
        patience: int,
        eval_roi_conf_thresh: float,
        eval_roi_max_dets: int,
        eval_nms_iou_thresh: float,
        eval_rcnn_conf_thresh: float,
        output_dir: str = "output",
    ) -> None:
        ### models
        self._im_to_x = make_input_transform(input_dims)
        self.model = model.cuda()

        ### datasets
        self.trn_loader = BatchLoader(trn_manager, batch_size)
        self.val_loader = BatchLoader(val_manager, batch_size)
        self.rep_manager = rep_manager
        self.batch_size = batch_size

        ### ground-truth label encoding
        anchors = build_anchors(
            (d // fm_stride for d in input_dims), anchor_areas, anchor_aspect_ratios
        )
        self._anchor_encoder = AnchorEncoder(
            anchors, encoder_iou_thresh, encoder_iou_margin
        )
        self._region_encoder = RegionEncoder(encoder_iou_thresh)
        self._region_filter = PredictionFilterPipeline(
            ConfidenceFilter(trn_roi_conf_thresh),
            MaxDetFilter(trn_roi_max_dets),
            NMSFilter(trn_roi_nms_iou_thresh),
        )  # filters rois before rcnn when training

        ### loss
        self._rpn_loss_func = RPNLoss(alpha, gamma).cuda()
        self._rcnn_loss_func = RCNNLoss(alpha, gamma).cuda()
        self._track_loss_func = TrackLoss().cuda()
        self._loss_coefs = torch.as_tensor(loss_coefs).cuda()

        ### optimizers
        self._optim = SGD(self.model.parameters(), **sgd_kwargs)

        self.patience = patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.detector = Detector(
            model,
            anchors,
            input_dims,
            eval_roi_conf_thresh,
            eval_roi_max_dets,
            eval_nms_iou_thresh,
            eval_rcnn_conf_thresh,
        )
        self.report_im_h = 200

        ### state
        self.n_iters = 0
        self.best_val_loss = float("inf")
        self.iters_no_improvement = 0

    def _forward_loss(self, instance: Tuple[ImageInstance, ImageInstance]) -> DTLoss:
        """compute joint loss for a single instance.

        Args:
            instance: (image, labels) tuple for time t, t+tau.

        Returns:
            dt_loss:
                o_loss: RPN binary classification loss.
                b_loss_rpn: RPN bounding box regression loss.
                c_loss: RCNN multiclass classification loss.
                b_loss_rcnn: RCNN bounding box regression loss.
                t_loss: cross-frame tracking loss.
        """
        inst_0, inst_1 = instance

        ### extract feature maps.
        x0 = self._im_to_x(inst_0.im)  # (3, H, W)
        x1 = self._im_to_x(inst_1.im)  # (3, H, W)
        x = torch.stack([x0, x1]).cuda()  # (2, 3, H, W)
        fmaps = self.model.backbone(x)  # pyramid of feature maps 3*(2, ...)

        ### compute losses for RPN
        ###   - inputs are feature maps
        ###   - supervision from ground-truth labels
        # RPN label encoding.
        lw0_rpn, c0_star_rpn, b0_star_rpn = self._anchor_encoder(inst_0.labels)
        lw1_rpn, c1_star_rpn, b1_star_rpn = self._anchor_encoder(inst_1.labels)
        lw_rpn = np.stack([lw0_rpn, lw1_rpn])  # (2, |A|)
        c_star_rpn = np.stack([c0_star_rpn, c1_star_rpn]) != 0  # (2, |A|)
        b_star_rpn = np.stack([b0_star_rpn, b1_star_rpn])  # (2, |A|, 4)
        # RPN predictions.
        o_hat_rpn, b_hat_rpn, fm_reg = self.model.rpn(fmaps["c4"])
        # RPN loss.
        lw_rpn = torch.as_tensor(lw_rpn, dtype=torch.float32).cuda()
        c_star_rpn = torch.as_tensor(c_star_rpn, dtype=torch.int64).cuda()
        b_star_rpn = torch.as_tensor(b_star_rpn, dtype=torch.float32).cuda()
        o_loss_rpn, b_loss_rpn = self._rpn_loss_func(
            lw_rpn, o_hat_rpn, c_star_rpn, b_hat_rpn, b_star_rpn
        )

        ### compute losses for RCNN
        ###   - inputs are feature maps and RPN output
        ###   - supervision from ground-truth labels and regions from RPN output
        # acquire filtered regions for ROI pooling.
        o0_hat_rpn, o1_hat_rpn = [
            tensor_to_ndarray(confs)
            for confs in o_hat_rpn[:, :, 1]  # confidence for "object" class
        ]  # 2 * (|A|,)
        rboxes_0, rboxes_1 = [
            frcnn_box_decode(
                self._anchor_encoder.anchors,  # (|A|, 4)
                tensor_to_ndarray(offsets),  # (|A|, 4)
            )  # (|A|, 4)
            for offsets in b_hat_rpn  # (2, |A|, 4)
        ]  # 2*(|A|, 4)
        _, rboxes_0 = self._region_filter(o0_hat_rpn, rboxes_0)  # (|R0|, 4)
        _, rboxes_1 = self._region_filter(o1_hat_rpn, rboxes_1)  # (|R1|, 4)
        # would prefer to have encoding details abstracted away by a dataset
        # object, but the 2-stage structure complicates this. the main issue
        # is that the (unencoded) ground truth labels are required again once
        # we have obtained the region proposals in order to encode the labels
        # for the rcnn.
        c0_star_rcnn, b0_star_rcnn = self._region_encoder(
            rboxes_0, inst_0.labels
        )  # (|R0|,), (|R0|, 4)
        c1_star_rcnn, b1_star_rcnn = self._region_encoder(
            rboxes_1, inst_1.labels
        )  # (|R1|,), (|R1|, 4)
        c_star_rcnn = np.concatenate([c0_star_rcnn, c1_star_rcnn])  # (|R0 u R1|,)
        b_star_rcnn = np.concatenate([b0_star_rcnn, b1_star_rcnn])  # (|R0 u R1|, 4)
        # RCNN predictions.
        c5_0, c5_1 = fmaps["c5"]  # 2*(C', H', W')
        rboxes_0 = torch.as_tensor(rboxes_0, dtype=torch.float32).cuda()  # (|R0|, 4)
        rboxes_1 = torch.as_tensor(rboxes_1, dtype=torch.float32).cuda()  # (|R1|, 4)
        c0_hat_rcnn, b0_hat_rcnn = self.model.rcnn(c5_0, rboxes_0)  # (|R0|, ...)
        c1_hat_rcnn, b1_hat_rcnn = self.model.rcnn(c5_1, rboxes_1)  # (|R1|, ...)
        c_hat_rcnn = torch.cat([c0_hat_rcnn, c1_hat_rcnn])  # (|R0 u R1|, n_classes)
        b_hat_rcnn = torch.cat([b0_hat_rcnn, b1_hat_rcnn])  # (|R0 u R1|, 4)
        # RCNN loss.
        c_star_rcnn = torch.as_tensor(
            c_star_rcnn, dtype=torch.int64
        ).cuda()  # (|R0 u R1|,)
        b_star_rcnn = torch.as_tensor(
            b_star_rcnn, dtype=torch.float32
        ).cuda()  # (|R0 u R1|, 4)
        c_loss_rcnn, b_loss_rcnn = self._rcnn_loss_func(
            c_hat_rcnn, c_star_rcnn, b_hat_rcnn, b_star_rcnn
        )

        ### compute losses for correlation trackers
        ###   - inputs are backbone and RPN feature maps from each time step
        ###   - supervision from ground-truth labels from each time step
        # CT label encoding.
        track_rois, t_star = track_encode(
            inst_0.labels, inst_1.labels
        )  # 2 * (|R0 n R1|, 4)
        # CT predictions.
        # start by unzipping features from each time step
        c3_0, c3_1 = fmaps["c3"]  # 2 * (C, H, W)
        c4_0, c4_1 = fmaps["c4"]  # 2 * (C, H', W')
        c5_0, c5_1 = fmaps["c5"]  # 2 * (C, H', W')
        fm_pyr0 = OrderedDict(c3=c3_0, c4=c4_0, c5=c5_0)
        fm_pyr1 = OrderedDict(c3=c3_1, c4=c4_1, c5=c5_1)
        fm_reg0, fm_reg1 = fm_reg  # 2 * (Cr, Hr, Wr) RPN feature maps
        track_rois = torch.as_tensor(
            track_rois, dtype=torch.float32
        ).cuda()  # (|R0 n R1|, 4)
        t_hat = self.model.c_tracker(
            fm_pyr0, fm_pyr1, fm_reg0, fm_reg1, track_rois
        )  # (|R0 n R1|, 4)
        # CT loss.
        t_star = torch.as_tensor(t_star, dtype=torch.float32).cuda()
        t_loss = self._track_loss_func(t_hat, t_star)

        dt_loss = DTLoss(
            o_loss=o_loss_rpn,
            b_loss_rpn=b_loss_rpn,
            c_loss=c_loss_rcnn,
            b_loss_rcnn=b_loss_rcnn,
            t_loss=t_loss,
        )

        return dt_loss

    def _minibatch_loss(
        self, minibatch: Sequence[Tuple[ImageInstance, ImageInstance]]
    ) -> DTLoss:
        """compute averaged loss for a single minibatch"""
        minibatch_loss = DTLoss(requires_grad=True)
        for instance in minibatch:
            minibatch_loss += self._forward_loss(instance)

        return minibatch_loss

    def train(self) -> DTLoss:
        self.model.train()
        trn_loss = DTLoss()
        for minibatch in self.trn_loader:
            minibatch_loss = self._minibatch_loss(minibatch)

            self._optim.zero_grad()
            minibatch_loss.backward(self._loss_coefs)
            self._optim.step()

            trn_loss += minibatch_loss
            self.n_iters += len(minibatch)

        return trn_loss

    @torch.no_grad()
    def validate(self) -> DTLoss:
        self.model.eval()
        val_loss = DTLoss()
        with torch.no_grad():
            for minibatch in self.val_loader:
                minibatch_loss = self._minibatch_loss(minibatch)

                val_loss += minibatch_loss

        return val_loss

    def _generate_report_labels(self, confs: np.ndarray, top_n: int) -> Sequence[str]:
        """convert confidences to readable labels, showing confidences
        for top_n classes."""
        top_classes = np.argsort(confs, axis=1)[:, ::-1][:, :top_n]  # (|D|, top_n)
        top_confs = np.take_along_axis(confs, top_classes, axis=1)  # (|D|, top_n)

        labels = list()
        for det_top_classes, det_top_confs in zip(top_classes, top_confs):
            det_label = "\n".join(
                [
                    f"{class_int}: {conf:.2f}"
                    for class_int, conf in zip(det_top_classes, det_top_confs)
                ]
            )
            labels.append(det_label)

        return labels

    def _resize_report_im(self, im: Image.Image) -> Image.Image:
        """resize image such that it has height of `self.report_im_h`
        without changing aspect ratio."""
        im_w, im_h = im.size
        resize_ratio = self.report_im_h / im_h
        new_h, new_w = (int(resize_ratio * dim) for dim in (im_h, im_w))

        resized = im.resize((new_w, new_h))

        return resized

    @torch.no_grad()
    def report(self, trn_loss: DTLoss, val_loss: DTLoss) -> None:
        """report training progress."""
        TOP_N = 3

        report_ims = dict()
        for report_idx, (i0, i1) in enumerate(self.rep_manager):
            confs0, confs1, bboxes0, bboxes1, tracks = self.detector(i0.im, i1.im)

            im0 = self._resize_report_im(i0.im)
            im1 = self._resize_report_im(i1.im)
            draw_detections(im0, bboxes0, self._generate_report_labels(confs0, TOP_N))
            draw_detections(im1, bboxes1, self._generate_report_labels(confs1, TOP_N))

            cat_arr = np.concatenate([np.array(im0), np.array(im1)], axis=1)
            cat_im = Image.fromarray(cat_arr)
            report_ims[f"pair_{report_idx}"] = wandb.Image(cat_im)

        trn_metrics = {f"trn_{k}": float(v) for k, v in trn_loss.asdict().items()}
        val_metrics = {f"val_{k}": float(v) for k, v in val_loss.asdict().items()}

        wandb.log({**trn_metrics, **val_metrics, **report_ims})
        print(" ".join([str(trn_loss), str(val_loss)]))

    def step(self) -> None:
        """train on subset, validate, and report."""
        trn_loss = self.train()
        val_loss = self.validate()
        self.report(trn_loss, val_loss)

        scalar_val_loss = float(val_loss.to_scalar(self._loss_coefs))
        if scalar_val_loss < self.best_val_loss:
            self.best_val_loss = scalar_val_loss
            self.iters_no_improvement = 0
            torch.save(self.model.state_dict(), Path(self.output_dir, "weights.pt"))
        else:
            self.iters_no_improvement += 1

    def run(self, max_iters: int = math.inf) -> None:
        """iterate until a stopping condition is satisfied."""
        while not (
            self.iters_no_improvement > self.patience or self.n_iters > max_iters
        ):
            self.step()
