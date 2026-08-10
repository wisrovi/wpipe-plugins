# WPipe Changelog

All notable changes to WPipe will be documented in this file.

---

## [wpipe-plugins 0.1.2] - 2026-08-10

### Fixed
- `draw_detections` now draws on a uint8 copy when given a float [0, 1] image
  and returns it in the same dtype/range. `cv2.putText` raises an assertion on
  float images since OpenCV 5.0.0, which broke detection and segmentation
  visualizations on fresh installs.
- Added `tests/vision/ecam_yolo/test_vision_helpers.py` covering float and
  uint8 inputs to `draw_detections`.

### Changed
- Package version bumped to 0.1.2 (patch release).

---

## [wpipe-plugins 0.1.1] - 2026-08-10

### Added
- `example_basic.py`: minimal ECAM-YOLO example with all defaults and `model_path` passed in the run
- `example_batch_images.py`, `example_directory.py`, `example_model_in_run.py`: batch, directory and run-only-model input modes
- Device auto-detection: `ECAMConfig.device` is `None` by default and resolves to GPU when available, otherwise CPU; an explicit value is used as-is
- Graceful degradation without model (`no_model`) or without images (`no_images`) instead of raising
- `wpipe-plugins/AGENTS.md` with repository contribution rules

### Changed
- `ImageECamYOLO.__call__` sets `save`, `verbose` and `image_name` internally (immutable run fields removed)
- `ECAMConfig.confidence_threshold` defaults to 0.25
- Examples import only from the installed library; `model_path` goes either in the config or in the run, never both
- Package version bumped to 0.1.1 (patch release)

---

## [1.5.1] - 2026-04-10

### Fixed
- Alert system API compatibility with new `expression` parameter
- Performance comparison example using `get_stats()` instead of deprecated method
- Reduced package size (42MB → 140KB) by excluding heavy examples

---

## [1.5.0] - 2026-04-10

### Added
- **ParallelExecutor**: Execute pipeline steps in parallel (ThreadPoolExecutor/ProcessPoolExecutor)
- **ExecutionMode**: IO_BOUND, CPU_BOUND, SEQUENTIAL
- **DAGScheduler**: Dependency graph management with topological sorting
- **PipelineAsStep**: Use pipelines as steps in other pipelines
- **@step()** decorator: Inline step definition
- **StepRegistry**: Central registry for decorated steps
- **ResourceMonitor**: Track RAM/CPU during execution
- **Exporter**: JSON/CSV export capabilities
- **Type validators**: Input/output validation

---

## [1.0.0] - 2024-04-01

### Added
- **Pipeline**: Core pipeline orchestration
- **Condition**: Conditional branching based on data
- **Retry**: Automatic retry with backoff
- **APIClient**: External API integration
- **SQLite/Wsqlite**: Data persistence
- **Error handling**: Custom exceptions with codes
- **YAML config**: Load configurations from YAML
- **Nested pipelines**: Compose complex workflows
- **Progress tracking**: Rich terminal output
- **Type hints**: Complete type annotations