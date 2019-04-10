"""training script."""

from argparse import ArgumentParser
from types import SimpleNamespace

import yaml
from ml_utils.prediction_filtering import (
    PredictionFilterPipeline,
    ConfidenceFilter,
    NMSFilter
)

from detect_to_track.models import DetectTrackModule
from detect_to_track.data.imagenet import ImagenetTrnManager, ImagenetValManager
from detect_to_track.data.encoding import AnchorEncoder, RegionEncoder
from detect_to_track.trainer import DetectTrackTrainer
from detect_to_track.utils import build_anchors


parser = ArgumentParser(__doc__)
parser.add_argument(
    '-c', '--cfg',
    default='cfg/default.yaml',
    help='path to cfg file'
)
args = parser.parse_args()

cfg = SimpleNamespace(**yaml.load(open(args.cfg)))


anchor_cfg = cfg.ANCHORS
n_anchors = len(anchor_cfg['scale_factors']) * len(anchor_cfg['aspect_ratios'])

model = DetectTrackModule()
model.build_backbone(cfg.DEPTH)
model.build_rpn(cfg.C4_CHANNELS, n_anchors)
model.build_rcnn(cfg.C5_CHANNELS, cfg.N_CLASSES, cfg.K)
model.build_c_tracker(cfg.REG_CHANNELS, cfg.D_MAX, cfg.R_HW)
model.backbone.freeze()  # needs to be done after state dict loaded

trn_set = ImagenetTrnManager(cfg.DATA_ROOT, cfg.P_DET)
val_set = ImagenetValManager(cfg.DATA_ROOT, cfg.VAL_SIZE)

anchors = build_anchors(cfg.C4_HW, **anchor_cfg)

encoding_cfg = cfg.ENCODING
anchor_encoder = AnchorEncoder(anchors, **encoding_cfg)
region_encoder = RegionEncoder(encoding_cfg['iou_thresh'])

region_filter = PredictionFilterPipeline(
    ConfidenceFilter(cfg.TRAIN_ROI_CONF_THRESH),
    NMSFilter(cfg.TRAIN_IOU_CONF_THRESH)
)

loss_cfg = cfg.LOSS
trainer = DetectTrackTrainer(
    model,
    trn_set,
    val_set,
    cfg.SPLIT_SIZE,
    cfg.BATCH_SIZE,
    cfg.NET_INPUT_HW,
    anchor_encoder,
    region_encoder,
    region_filter,
    loss_cfg['alpha'],
    loss_cfg['gamma'],
    loss_cfg['coefs'],
    cfg.OPTIM,
    cfg.PATIENCE
)

trainer.train()
