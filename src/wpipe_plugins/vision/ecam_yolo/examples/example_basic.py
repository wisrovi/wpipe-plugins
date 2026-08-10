"""Minimal ECAM-YOLO example: defaults everywhere, model passed in the run.

Usage:
    python example_basic.py --model yolov8n.pt --image bus.jpg
"""

import argparse
import os
from typing import Any

from loguru import logger
from wpipe import Pipeline

from wpipe_plugins.vision.ecam_yolo import ECAMConfig, ImageECamYOLO


def run_basic_example(
    model_path: str,
    image_path: str,
    output_dir: str = "./output/basic",
) -> None:
    """Run inference with defaults; the model is supplied in the run data.

    Args:
        model_path (str): Path to the YOLO weights file (``.pt``).
        image_path (str): Path to the input image.
        output_dir (str): Directory where the visualization is saved.
    """
    step = ImageECamYOLO(ECAMConfig())

    pipe = Pipeline(pipeline_name="basic_example_pipeline", verbose=False)
    pipe.set_steps([step])

    os.makedirs(output_dir, exist_ok=True)
    inference_data = {
        "image_data": image_path,
        "model_path": model_path,
        "output_dir": output_dir,
    }

    try:
        context: dict[str, Any] = pipe.run(inference_data)
        for result in context.get("results", []):
            detections = [
                obj for obj in result["model_results"] if obj.get("status") == "ok"
            ]
            logger.info(f" - {len(detections)} detection(s)")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal YOLO + Eigen-CAM example (wpipe).")
    parser.add_argument("--model", required=True, help="Path to YOLO weights file (.pt)")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument(
        "--output-dir",
        default="./output/basic",
        help="Directory to save the visualization",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_basic_example(args.model, args.image, args.output_dir)
