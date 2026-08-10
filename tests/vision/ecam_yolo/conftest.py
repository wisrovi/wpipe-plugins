"""Shared fixtures for the ECAM-YOLO example tests.

The real inference stack (ultralytics YOLO + pytorch_grad_cam) requires model
weights and a GPU, so these tests stand them in with light fakes that expose
the same surface the ``ImageECamYOLO`` step consumes. This lets us run each
example end-to-end through a real ``wpipe`` pipeline without artifacts.
"""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

ECAM_MODULE = "wpipe_plugins.vision.ecam_yolo.states.ecam_yolo"

pytest.importorskip("wpipe")
pytest.importorskip("torch")
pytest.importorskip("ultralytics")
pytest.importorskip("pytorch_grad_cam")


class _FakeDetModel(nn.Module):
    """Minimal model exposing the attributes YOLO model audit expects."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.Conv2d(16, 32, 3))


class _FakeBox:
    def __init__(self) -> None:
        self.conf = np.array([0.9])
        self.cls = np.array([0])
        self.xyxy = torch.tensor([[30.0, 30.0, 150.0, 150.0]])


class _FakeMasks:
    """Ultralytics ``Masks`` expose both ``xy`` and ``__len__``."""

    def __init__(self) -> None:
        self.xy = [
            np.array(
                [[40.0, 40.0], [100.0, 40.0], [100.0, 100.0], [40.0, 100.0]],
                dtype=np.float32,
            )
        ]

    def __len__(self) -> int:
        return len(self.xy)


class _FakeResult:
    def __init__(self, task: str, names: dict[int, str]) -> None:
        self.names = names
        self.orig_img = np.zeros((200, 200, 3), np.uint8)
        self.path = "/media/sample_image.jpg"
        self.task = task
        if task == "classify":
            self.probs = SimpleNamespace(top1=0, top1conf=0.92)
            self.boxes = None
            self.masks = None
        else:
            self.probs = None
            self.boxes = [_FakeBox()]
            self.masks = None if task == "detect" else _FakeMasks()


class FakeYOLO:
    """Stand-in for ``ultralytics.YOLO`` that needs no weights file."""

    def __init__(self, model_path: str) -> None:
        """Infer the fake task type from the model file stem."""
        stem = Path(model_path).stem
        if "cls" in stem:
            self.task = "classify"
        elif "seg" in stem:
            self.task = "segment"
        else:
            self.task = "detect"
        self.model = _FakeDetModel()
        self.names = {0: "person"}

    def predict(self, source=None, conf=None, device=None, verbose=False, imgsz=None):
        """Return a single fake inference result for the inferred task."""
        return [_FakeResult(self.task, self.names)]


class FakeEigenCAM:
    """Stand-in for ``pytorch_grad_cam.EigenCAM`` returning a constant heatmap."""

    def __init__(self, *args, **kwargs) -> None:
        """Accept any Grad-CAM constructor arguments."""
        pass

    def __call__(self, x):
        """Return a constant centered heatmap matching the input shape."""
        h, w = x.shape[2], x.shape[3]
        cam = np.zeros((h, w), np.float32)
        cam[80:120, 80:120] = 0.8
        return cam[None]


@pytest.fixture
def ecam_env(monkeypatch, tmp_path):
    """Swap the heavy inference classes for fakes and run inside a temp dir."""
    module = importlib.import_module(ECAM_MODULE)
    monkeypatch.setattr(module, "YOLO", FakeYOLO)
    monkeypatch.setattr(module, "EigenCAM", FakeEigenCAM)
    monkeypatch.chdir(tmp_path)
    return module
