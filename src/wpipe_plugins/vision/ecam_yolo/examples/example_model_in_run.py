"""Example: pass the model in the run data instead of the step config.

Usage:
    python example_model_in_run.py --model yolov8n.pt --image bus.jpg
                                   [--output-dir ./output/model-in-run]
                                   [--device cpu] [--conf 0.25]
"""

import argparse
import os
from typing import Any

from loguru import logger
from wpipe import Pipeline

# Importing from the installed library
from wpipe_plugins.vision.ecam_yolo import ECAMConfig, ImageECamYOLO


def run_model_in_run_example(
    model_path: str,
    image_path: str,
    output_dir: str = "./output/model-in-run",
    device: str = "cpu",
    confidence_threshold: float = 0.25,
) -> None:
    """Run inference with the model supplied only via the run data.

    The step is created without a model (``ECAMConfig()``); ``model_path`` in
    the run data loads it at execution time.

    Args:
        model_path (str): Path to the YOLO weights file (``.pt``).
        image_path (str): Path to the input image.
        output_dir (str): Directory where the visualization is saved.
        device (str): Compute device, ``'cpu'`` or a GPU index like ``'0'``.
        confidence_threshold (float): Minimum confidence for a valid prediction.
    """
    # 1. Setup Configuration WITHOUT a model: it is provided in the run data.
    config = ECAMConfig(
        confidence_threshold=confidence_threshold,
        device=device,
    )

    # 2. Initialize the ECAM Step (no model loaded yet)
    model_in_run_step = ImageECamYOLO(config)

    # 3. Setup the Pipeline
    pipe = Pipeline(pipeline_name="model_in_run_example_pipeline", verbose=True)
    pipe.set_steps([model_in_run_step])

    # 4. Prepare Input Data: "model_path" here supplies the model for this run.
    os.makedirs(output_dir, exist_ok=True)

    inference_data = {
        "image_data": image_path,
        "model_path": model_path,
        "output_dir": output_dir,
    }

    # 5. Run the Pipeline
    logger.info("Starting pipeline with the model provided in the run data...")
    try:
        context: dict[str, Any] = pipe.run(inference_data)
        results: list[dict[str, Any]] = context.get("results", [])

        for result in results:
            detections = [
                obj for obj in result["model_results"] if obj.get("status") == "ok"
            ]
            logger.info(f" - {len(detections)} detection(s)")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO + Eigen-CAM with the model supplied in the run data (wpipe)."
    )
    parser.add_argument("--model", required=True, help="Path to YOLO weights file (.pt)")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument(
        "--output-dir",
        default="./output/model-in-run",
        help="Directory to save the visualization",
    )
    parser.add_argument(
        "--device", default="cpu", help="Compute device: 'cpu' or a GPU index like '0'"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_model_in_run_example(args.model, args.image, args.output_dir, args.device, args.conf)
