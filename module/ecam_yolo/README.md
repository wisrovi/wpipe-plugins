# ECAM-YOLO Plugin

The **ECAM-YOLO** plugin provides a seamless integration of YOLO models with **Eigen-CAM** visualization within the `wpipe` pipeline ecosystem. It automatically supports object detection, image classification, and segmentation tasks, generating heatmaps that highlight the regions influencing model predictions.

## Key Features

- **Automated Task Detection**: Automatically identifies if the loaded model is for classification, detection, or segmentation.
- **Eigen-CAM Integration**: Generates spatial heatmaps to visualize model focus.
- **Dynamic Layer Selection**: Automatically identifies the optimal feature map layers for CAM across different YOLO architectures.
- **Multi-task Support**: 
  - **Classification**: Highlights features relevant to the top-1 class.
  - **Detection**: Renormalizes heatmaps within detected bounding boxes.
  - **Segmentation**: Blends heatmaps with translucent masks.
- **Seamless wpipe Integration**: Implemented as a `wpipe` step for easy pipeline inclusion.

## Plugin Structure

```text
ecam_yolo/
├── config/             # Configuration constants (DEVICE, etc.)
├── examples/           # Task-specific usage examples
├── exceptions/         # Custom YoloError exceptions
├── schemas/            # Pydantic data models (InferenceObj)
├── states/             # Core logic (ImageECamYOLO, ECAMConfig)
├── utils/              # Helper functions for parsing and rendering
└── wrappers/           # YOLO model standardization for CAM compatibility
```

## Core Components

### `ECAMConfig`
The configuration class used to initialize the engine.
- `model_path`: Path to the YOLO weights (`.pt`).
- `confidence_threshold`: Minimum confidence for valid predictions.
- `device`: Compute device (`cpu` or GPU index).

### `ImageECamYOLO` (The Step)
The main execution engine. It's decorated as a `wpipe` step and can be called directly within a pipeline. It handles:
1. **Inference**: Running the YOLO model.
2. **CAM Generation**: Using Eigen-CAM to extract feature importance.
3. **Task Processing**: Rendering results based on the detected task type.
4. **Visualization**: Saving and returning the final heatmaps.

## Usage

### 1. Basic Setup
```python
from wpipe_plugins.vision.ecam_yolo import ECAMConfig, ImageECamYOLO
from wpipe import Pipeline

# Configure the engine
config = ECAMConfig(
    model_path="/path/to/model.pt",
    confidence_threshold=0.3,
    device="cpu"
)

# Initialize the step
ecam_step = ImageECamYOLO(config)

# Setup the pipeline
pipe = Pipeline(pipeline_name="my_pipeline")
pipe.set_steps([ecam_step])
```

### 2. Running Inference
The step expects an input dictionary (or an `InferenceObj`) containing:
- `image_data`: Path to the image, PIL Image, or Numpy array.
- `output_dir`: Directory to save visualizations.

```python
inference_data = {
    "image_data": "sample.jpg",
    "output_dir": "./output",
    "verbose": True
}

results = pipe.run(inference_data)
```

## Task-Specific Examples
Refer to the `examples/` directory for detailed standalone scripts for:
- `example_classification.py`
- `example_detection.py`
- `example_segmentation.py`

## Requirements
- `ultralytics`
- `pytorch-grad-cam`
- `opencv-python`
- `loguru`
- `wpipe`
