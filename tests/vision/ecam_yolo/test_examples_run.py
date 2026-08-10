"""End-to-end tests for the ECAM-YOLO example pipelines.

Each example must run its full pipeline (config -> step -> wpipe Pipeline ->
inference -> visualization) and consume the results without failing. The heavy
stack (YOLO weights + EigenCAM) is faked via the ``ecam_env`` fixture; the
wpipe engine and the plugin's own rendering code run for real.
"""

import importlib
from pathlib import Path

import pytest
from wpipe import Pipeline

from wpipe_plugins.vision.ecam_yolo import ECAMConfig, ImageECamYOLO

EXAMPLES = {
    "detection": {
        "module": "wpipe_plugins.vision.ecam_yolo.examples.example_detection",
        "runner": "run_detection_example",
        "model": "/models/yolo26l/yolo26l-detec.pt",
        "expected_output": Path("output") / "detection" / "sample_image.jpg",
    },
    "classification": {
        "module": "wpipe_plugins.vision.ecam_yolo.examples.example_classification",
        "runner": "run_classification_example",
        "model": "/models/yolo26l/yolo26l-cls.pt",
        "expected_output": Path("output") / "classification" / "sample_image.jpg",
    },
    "segmentation": {
        "module": "wpipe_plugins.vision.ecam_yolo.examples.example_segmentation",
        "runner": "run_segmentation_example",
        "model": "/models/yolo26l/yolo26l-seg.pt",
        "expected_output": Path("output") / "segmentation" / "sample_image.jpg",
    },
}


@pytest.mark.parametrize("name", list(EXAMPLES), scope="function")
def test_example_runs_end_to_end(name, ecam_env, capfd):
    """The example must run its pipeline and produce a non-empty visualization."""
    info = EXAMPLES[name]
    runner = getattr(importlib.import_module(info["module"]), info["runner"])

    runner(
        model_path=info["model"],
        image_path="/media/sample_image.jpg",
    )

    # The visualization file must exist and be non-empty.
    output = info["expected_output"]
    assert output.exists(), f"expected visualization at {output}"
    assert output.stat().st_size > 0

    # Examples wrap everything in try/except; make sure nothing failed silently.
    captured = capfd.readouterr()
    assert "Pipeline execution failed" not in captured.err
    assert "Pipeline execution failed" not in captured.out


@pytest.mark.parametrize("name", list(EXAMPLES), scope="function")
def test_step_result_is_mergeable_and_parsed(name, ecam_env):
    """The step must return a mergeable dict whose ``results`` hold parsed objects.

    The wpipe merge contract requires a dict return; ``results`` must contain
    one entry per image with parsed ``model_results``.
    """
    info = EXAMPLES[name]
    config = ECAMConfig(model_path=info["model"], confidence_threshold=0.3, device="cpu")
    step = ImageECamYOLO(config)

    pipe = Pipeline(pipeline_name=f"{name}_pipeline_test", verbose=False)
    pipe.set_steps([step])

    context = pipe.run(
        {
            "image_data": "/media/sample_image.jpg",
            "output_dir": "./output/test",
        }
    )

    assert isinstance(context, dict)
    results = context.get("results")
    assert isinstance(results, list)
    assert len(results) == 1

    model_results = results[0]["model_results"]
    assert isinstance(model_results, list)
    assert len(model_results) >= 1
    assert model_results[0]["status"] == "ok"


def _run_with_device_spy(ecam_env, monkeypatch, device: str | int | None) -> list:
    """Run the step and return the list of ``device`` values passed to predict."""
    step = ImageECamYOLO(
        ECAMConfig(
            model_path="/models/yolo26l/yolo26l-detec.pt",
            confidence_threshold=0.3,
            device=device,
        )
    )
    calls: list = []
    original_predict = ecam_env.YOLO.predict

    def spy(self, source=None, conf=None, device=None, verbose=False, imgsz=None):
        calls.append(device)
        return original_predict(
            self, source=source, conf=conf, device=device, verbose=verbose, imgsz=imgsz
        )

    monkeypatch.setattr(ecam_env.YOLO, "predict", spy)

    pipe = Pipeline(pipeline_name="device_test", verbose=False)
    pipe.set_steps([step])
    pipe.run(
        {
            "image_data": "/media/sample_image.jpg",
            "output_dir": "./output/test",
        }
    )
    return calls


def test_device_autodetects_gpu_when_available(ecam_env, monkeypatch):
    """Without an explicit device, CUDA is used when available."""
    monkeypatch.setattr(ecam_env.torch.cuda, "is_available", lambda: True)
    calls = _run_with_device_spy(ecam_env, monkeypatch, device=None)
    assert calls == [0]


def test_device_autodetects_cpu_when_no_gpu(ecam_env, monkeypatch):
    """Without an explicit device, CPU is the fallback when CUDA is missing."""
    monkeypatch.setattr(ecam_env.torch.cuda, "is_available", lambda: False)
    calls = _run_with_device_spy(ecam_env, monkeypatch, device=None)
    assert calls == ["cpu"]


def test_explicit_device_is_used_without_gpu_check(ecam_env, monkeypatch):
    """An explicit device in the config is used as-is, ignoring GPU availability."""
    monkeypatch.setattr(ecam_env.torch.cuda, "is_available", lambda: False)
    calls = _run_with_device_spy(ecam_env, monkeypatch, device="cuda:2")
    assert calls == ["cuda:2"]


def test_run_without_model_returns_graceful_error(ecam_env):
    """A step without a model (config nor run) must not break the pipeline.

    Instead of raising, it returns a ``no_model`` result so the run continues.
    """
    config = ECAMConfig(confidence_threshold=0.3, device="cpu")
    step = ImageECamYOLO(config)
    assert step.yolo_model is None

    pipe = Pipeline(pipeline_name="no_model_test", verbose=False)
    pipe.set_steps([step])

    context = pipe.run(
        {
            "image_data": "/media/sample_image.jpg",
            "output_dir": "./output/test",
        }
    )

    assert isinstance(context, dict)
    results = context.get("results")
    assert isinstance(results, list)
    assert len(results) == 1

    model_results = results[0]["model_results"]
    assert isinstance(model_results, list)
    assert len(model_results) == 1
    assert model_results[0]["status"] == "no_model"
    assert "model" in model_results[0]["message"].lower()


def test_run_with_list_of_images(ecam_env):
    """A list of images in ``image_data`` yields one result per image."""
    config = ECAMConfig(
        model_path="/models/yolo26l/yolo26l-detec.pt",
        confidence_threshold=0.3,
        device="cpu",
    )
    step = ImageECamYOLO(config)

    pipe = Pipeline(pipeline_name="batch_test", verbose=False)
    pipe.set_steps([step])

    context = pipe.run(
        {
            "image_data": ["/media/a.jpg", "/media/b.jpg"],
            "output_dir": "./output/test",
        }
    )

    results = context.get("results")
    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        model_results = result["model_results"]
        assert isinstance(model_results, list)
        assert len(model_results) == 1
        assert model_results[0]["status"] == "ok"


def test_run_with_directory(ecam_env):
    """A directory in ``image_data`` is expanded to one result per image file."""
    (Path.cwd() / "img_a.jpg").write_bytes(b"fake")
    (Path.cwd() / "img_b.png").write_bytes(b"fake")
    (Path.cwd() / "notes.txt").write_bytes(b"fake")

    config = ECAMConfig(
        model_path="/models/yolo26l/yolo26l-detec.pt",
        confidence_threshold=0.3,
        device="cpu",
    )
    step = ImageECamYOLO(config)

    pipe = Pipeline(pipeline_name="directory_test", verbose=False)
    pipe.set_steps([step])

    context = pipe.run(
        {
            "image_data": str(Path.cwd()),
            "output_dir": "./output/test",
        }
    )

    results = context.get("results")
    assert isinstance(results, list)
    assert len(results) == 2  # only the images, not notes.txt


def test_run_with_empty_directory_is_graceful(ecam_env):
    """A directory without images returns a ``no_images`` result, no crash."""
    config = ECAMConfig(
        model_path="/models/yolo26l/yolo26l-detec.pt",
        confidence_threshold=0.3,
        device="cpu",
    )
    step = ImageECamYOLO(config)

    pipe = Pipeline(pipeline_name="empty_dir_test", verbose=False)
    pipe.set_steps([step])

    context = pipe.run(
        {
            "image_data": str(Path.cwd()),
            "output_dir": "./output/test",
        }
    )

    results = context.get("results")
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["model_results"][0]["status"] == "no_images"


def test_model_path_can_be_overridden_at_run_time(ecam_env):
    """A ``model_path`` in the run data overrides the model from the step config."""
    config = ECAMConfig(
        model_path="/models/yolo26l/yolo26l-detec.pt",
        confidence_threshold=0.3,
        device="cpu",
    )
    step = ImageECamYOLO(config)
    assert step.task == "detect"

    pipe = Pipeline(pipeline_name="override_model_test", verbose=False)
    pipe.set_steps([step])

    context = pipe.run(
        {
            "image_data": "/media/sample_image.jpg",
            "model_path": "/models/yolo26l/yolo26l-cls.pt",
            "output_dir": "./output/test",
        }
    )

    assert step.task == "classify"
    assert step.config.model_path == "/models/yolo26l/yolo26l-cls.pt"

    results = context.get("results")
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["model_results"][0]["status"] == "ok"


def test_model_can_be_provided_only_at_run_time(ecam_env):
    """An ``ECAMConfig`` without a model is valid if the run data supplies it."""
    config = ECAMConfig(confidence_threshold=0.3, device="cpu")
    step = ImageECamYOLO(config)
    assert step.yolo_model is None

    pipe = Pipeline(pipeline_name="run_only_model_test", verbose=False)
    pipe.set_steps([step])

    context = pipe.run(
        {
            "image_data": "/media/sample_image.jpg",
            "model_path": "/models/yolo26l/yolo26l-seg.pt",
            "output_dir": "./output/test",
        }
    )

    assert step.task == "segment"
    assert step.config.model_path == "/models/yolo26l/yolo26l-seg.pt"

    results = context.get("results")
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["model_results"][0]["status"] == "ok"


def _example_runner_info(name: str):
    """Return (module, runner, expected output, runner kwargs) for a new example."""
    return {
        "batch_images": (
            "wpipe_plugins.vision.ecam_yolo.examples.example_batch_images",
            "run_batch_images_example",
            Path("output") / "batch" / "sample_image.jpg",
            {"image_paths": ["/media/a.jpg", "/media/b.jpg"]},
        ),
        "directory": (
            "wpipe_plugins.vision.ecam_yolo.examples.example_directory",
            "run_directory_example",
            Path("output") / "directory" / "sample_image.jpg",
            {"images_dir": str(Path.cwd())},
        ),
        "model_in_run": (
            "wpipe_plugins.vision.ecam_yolo.examples.example_model_in_run",
            "run_model_in_run_example",
            Path("output") / "model-in-run" / "sample_image.jpg",
            {"image_path": "/media/sample_image.jpg"},
        ),
        "basic": (
            "wpipe_plugins.vision.ecam_yolo.examples.example_basic",
            "run_basic_example",
            Path("output") / "basic" / "sample_image.jpg",
            {"image_path": "/media/sample_image.jpg"},
        ),
    }[name]


@pytest.mark.parametrize(
    "name", ["batch_images", "directory", "model_in_run", "basic"]
)
def test_new_examples_run_end_to_end(name, ecam_env, capfd):
    """The new input-mode examples must run and save a non-empty visualization."""
    module_path, runner_name, expected_output, runner_kwargs = _example_runner_info(name)

    if name == "directory":
        (Path.cwd() / "a.jpg").write_bytes(b"fake")
        (Path.cwd() / "b.jpg").write_bytes(b"fake")

    runner = getattr(importlib.import_module(module_path), runner_name)
    runner(model_path="/models/yolo26l/yolo26l-detec.pt", **runner_kwargs)

    assert expected_output.exists(), f"expected visualization at {expected_output}"
    assert expected_output.stat().st_size > 0

    captured = capfd.readouterr()
    assert "Pipeline execution failed" not in captured.err
    assert "Pipeline execution failed" not in captured.out
