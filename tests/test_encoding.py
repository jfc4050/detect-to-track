import pytest
import numpy as np

from detect_to_track.data.encoding import frcnn_box_encode


@pytest.mark.parametrize("n_anchors", [0, 1, 2])
def test_frcnn_box_encode_handles_variable_anchors(n_anchors):
    anchors = np.random.rand(n_anchors, 4)
    boxes = np.random.rand(n_anchors, 4)

    offsets = frcnn_box_encode(anchors, boxes)

    assert offsets.shape == (n_anchors, 4)


@pytest.mark.parametrize("n_anchors", [0, 1, 2])
def test_frcnn_box_decode_handles_variable_anchors(n_anchors):
    anchors = np.random.rand(n_anchors, 4)
    offsets = np.random.rand(n_anchors, 4)

    boxes = frcnn_box_encode(anchors, offsets)

    assert boxes.shape == (n_anchors, 4)
