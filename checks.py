"""simple sanity checks"""

from argparse import ArgumentParser
from pathlib import Path


def check_sampler():
    from detect_to_track.data.imagenet import VIDTrnSampler, DETSampler

    root = Path("~/datasets/ILSVRC2015").expanduser()

    s = VIDTrnSampler(root)
    # s = DETSampler(root)

    i0, i1 = s.sample()

    i0.im.show()
    print(i0.labels)
    print()
    print(i1.labels)


def check_anchors():
    from detect_to_track.utils import build_anchors

    anchors = build_anchors(
        fm_shape=(2, 2), anchor_areas=[1, 2, 3], aspect_ratios=[1, 2], flatten=False
    )
    print(anchors)


if __name__ == "__main__":
    parser = ArgumentParser(__doc__)
    parser.add_argument("--sampler", action="store_true")
    parser.add_argument("--anchors", action="store_true")
    args = parser.parse_args()

    if args.sampler:
        check_sampler()
    if args.anchors:
        check_anchors()
