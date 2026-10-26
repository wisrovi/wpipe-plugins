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


def test_parse_detections_with_confidence_threshold():
    """Validates that parse_detections properly filters bounding boxes according to conf_threshold.

    This test creates a mock Ultralytics Results object containing two detections:
    one with low confidence (0.15) and one with high confidence (0.85).
    It verifies that when conf_threshold=0.2 is applied, the low-confidence detection is ignored,
    and when conf_threshold=0.1 is applied, both detections are successfully extracted.
    """
    import torch
    from types import SimpleNamespace
    from wpipe_plugins.vision.ecam_yolo.utils.vision_helpers import parse_detections

    class MockBox:
        def __init__(self, conf, cls, xyxy):
            self.conf = np.array([conf])
            self.cls = np.array([cls])
            self.xyxy = torch.tensor([xyxy])

    mock_res = SimpleNamespace(
        boxes=[
            MockBox(0.15, 0, [10.0, 10.0, 50.0, 50.0]),
            MockBox(0.85, 0, [60.0, 60.0, 100.0, 100.0]),
        ],
        names={0: "person"},
    )

    # Filtering with 0.2 should return only the high confidence detection
    boxes, colors, names = parse_detections([mock_res], conf_threshold=0.2)
    assert len(boxes) == 1
    assert np.array_equal(boxes[0], np.array([60, 60, 100, 100]))

    # Filtering with 0.1 should return both detections
    boxes_all, _, _ = parse_detections([mock_res], conf_threshold=0.1)
    assert len(boxes_all) == 2
