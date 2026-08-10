"""Smoke tests for the ECAM-YOLO example modules.

Every example must import cleanly as a package submodule and expose its
runnable entry point. Importing the example also pulls the whole plugin stack
(wpipe, torch, ultralytics, pytorch_grad_cam), so a green import proves the
dependency graph and the plugin packaging are coherent.
"""

import importlib

import pytest

EXAMPLES = {
    "detection": (
        "wpipe_plugins.vision.ecam_yolo.examples.example_detection",
        "run_detection_example",
    ),
    "classification": (
        "wpipe_plugins.vision.ecam_yolo.examples.example_classification",
        "run_classification_example",
    ),
    "segmentation": (
        "wpipe_plugins.vision.ecam_yolo.examples.example_segmentation",
        "run_segmentation_example",
    ),
    "batch_images": (
        "wpipe_plugins.vision.ecam_yolo.examples.example_batch_images",
        "run_batch_images_example",
    ),
    "directory": (
        "wpipe_plugins.vision.ecam_yolo.examples.example_directory",
        "run_directory_example",
    ),
    "model_in_run": (
        "wpipe_plugins.vision.ecam_yolo.examples.example_model_in_run",
        "run_model_in_run_example",
    ),
    "basic": (
        "wpipe_plugins.vision.ecam_yolo.examples.example_basic",
        "run_basic_example",
    ),
}


@pytest.mark.parametrize(("module_path", "runner"), EXAMPLES.values(), ids=list(EXAMPLES))
def test_example_imports_and_exposes_runner(module_path, runner):
    """The example module must import and expose its runnable entry point."""
    module = importlib.import_module(module_path)
    assert callable(getattr(module, runner))
