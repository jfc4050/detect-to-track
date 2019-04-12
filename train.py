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

prediction_map_hw = (int(x/16) for x in cfg.INPUT_SHAPE)
# TODO - move these somewhere else
C4_CHANNELS = 2048
C5_CHANNELS = 512
REG_CHANNELS = 512

### model setup
model = DetectTrackModule()
model.build_backbone(cfg.DEPTH)
model.build_rpn(
    C4_CHANNELS,
    len(cfg.ANCHOR_SCALE_FACTORS) * len(cfg.ANCHOR_ASPECT_RATIOS)
)
model.build_rcnn(C5_CHANNELS, cfg.N_CLASSES, cfg.K)
model.build_c_tracker(REG_CHANNELS, cfg.D_MAX, cfg.K)
model.backbone.freeze()  # needs to be done after state dict loaded

### anchor setup
anchors = build_anchors(
    prediction_map_hw, cfg.ANCHOR_SCALE_FACTORS, cfg.ANCHOR_ASPECT_RATIOS
)

### data setup
trn_set = ImagenetTrnManager(cfg.DATA_ROOT, cfg.P_DET)
val_set = ImagenetValManager(cfg.DATA_ROOT, cfg.VAL_SIZE)

### encoder setup
region_filter = PredictionFilterPipeline(
    ConfidenceFilter(cfg.TRAIN_ROI_CONF_THRESH),
    NMSFilter(cfg.TRAIN_NMS_IOU_THRESH)
)

trainer = DetectTrackTrainer(
    model,
    trn_set, val_set, cfg.SPLIT_SIZE, cfg.BATCH_SIZE,
    cfg.INPUT_SHAPE,
    anchors,
    cfg.ENCODER_IOU_THRESH, cfg.ENCODER_IOU_MARGIN,
    region_filter,
    cfg.ALPHA, cfg.GAMMA, cfg.COEFS,
    cfg.SGD_KWARGS,
    cfg.PATIENCE
)

trainer.train()
