"""training script."""

from argparse import ArgumentParser
from types import SimpleNamespace

import yaml

from detect_to_track.models import DetectTrackModule
from detect_to_track.data.imagenet import setup_vid_datasets
from detect_to_track.trainer import DetectTrackTrainer


parser = ArgumentParser(__doc__)
parser.add_argument("-c", "--cfg", default="cfg/default.yaml", help="path to cfg file")
args = parser.parse_args()
cfg = SimpleNamespace(**yaml.load(open(args.cfg)))

### model setup
model = DetectTrackModule()
model.build_backbone(cfg.BACKBONE_ARCH, cfg.FIRST_TRAINABLE_STAGE)
model.build_rpn(len(cfg.ANCHOR_AREAS) * len(cfg.ANCHOR_ASPECT_RATIOS))
model.build_rcnn(cfg.N_CLASSES, cfg.K)
model.build_c_tracker(cfg.D_MAX, cfg.K)
model.backbone.freeze()  # needs to be done after state dict loaded

trn_manager, val_manager = setup_vid_datasets(
    cfg.DATA_ROOT, cfg.VID_PARTITION_SIZES, cfg.TRN_SIZE, cfg.VAL_SIZE, cfg.P_DET, cfg.A
)

trainer = DetectTrackTrainer(
    model,
    trn_manager,
    val_manager,
    cfg.BATCH_SIZE,
    cfg.INPUT_SHAPE,
    cfg.FM_STRIDE,
    cfg.ANCHOR_AREAS,
    cfg.ANCHOR_ASPECT_RATIOS,
    cfg.ENCODER_IOU_THRESH,
    cfg.ENCODER_IOU_MARGIN,
    cfg.TRAIN_ROI_CONF_THRESH,
    cfg.TRAIN_MAX_ROIS,
    cfg.TRAIN_NMS_IOU_THRESH,
    cfg.ALPHA,
    cfg.GAMMA,
    cfg.COEFS,
    cfg.SGD_KWARGS,
    cfg.PATIENCE,
)

trainer.train()
