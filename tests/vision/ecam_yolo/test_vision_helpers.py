"""Unit tests for the ECAM-YOLO OpenCV rendering helpers."""

import numpy as np

from wpipe_plugins.vision.ecam_yolo.utils.vision_helpers import draw_detections


def test_draw_detections_accepts_float_image():
    """Float [0, 1] images must be drawn on a uint8 copy and returned as float.

    Since OpenCV 5.0.0, ``putText`` raises an assertion error on float images;
    the helper must convert internally and preserve the input dtype/range.
    """
    img = np.zeros((100, 100, 3), dtype=np.float32)
    boxes = [np.array([10, 10, 60, 60])]
    colors = [[0, 255, 0]]
    names = ["person"]

    out = draw_detections(boxes, colors, names, img)

    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert np.any(out > 0)


def test_draw_detections_preserves_uint8_input():
    """uint8 images are drawn in place and returned unchanged in dtype."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [np.array([10, 10, 60, 60])]
    colors = [[0, 255, 0]]
    names = ["person"]

    out = draw_detections(boxes, colors, names, img)

    assert out.dtype == np.uint8
    assert out.max() <= 255
    assert np.any(out > 0)
