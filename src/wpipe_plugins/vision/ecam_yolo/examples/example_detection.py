"""Example script for YOLO object detection with ECAM visualization.

Usage:
    python example_detection.py --model yolov8n.pt --image bus.jpg
                                [--output-dir ./output/detection]
                                [--device cpu] [--conf 0.25]
"""

import argparse
import os
from typing import Any

from loguru import logger
from wpipe import Pipeline

# Importing from the installed library
from wpipe_plugins.vision.ecam_yolo import ECAMConfig, ImageECamYOLO


def run_detection_example(
    model_path: str,
    image_path: str,
    output_dir: str = "./output/detection",
    device: str = "cpu",
    confidence_threshold: float = 0.25,
) -> None:
    """Demonstrates how to run object detection with ECAM.

    Args:
        model_path (str): Path to the YOLO weights file (``.pt``).
        image_path (str): Path to the input image.
        output_dir (str): Directory where the visualization is saved.
        device (str): Compute device, ``'cpu'`` or a GPU index like ``'0'``.
        confidence_threshold (float): Minimum confidence for a valid prediction.
    """
    pipe = Pipeline(pipeline_name="detection_example_pipeline", verbose=True)
    pipe.set_steps(
        [
            ImageECamYOLO(
                ECAMConfig(
                    model_path=model_path,
                    confidence_threshold=confidence_threshold,
                    device=device,
                )
            )
        ]
    )

    # 4. Prepare Input Data
    os.makedirs(output_dir, exist_ok=True)

    inference_data = {
        "image_data": image_path,
        "output_dir": output_dir,
    }

    # 5. Run the Pipeline
    logger.info("Starting detection pipeline...")
    try:
        context: dict[str, Any] = pipe.run(inference_data)
        results: list[dict[str, Any]] = context.get("results", [])

        for result in results:
            logger.info(f"Detected Objects: {len(result['model_results'])}")
            for obj in result["model_results"]:
                if obj.get("status") == "ok":
                    logger.info(f" - {obj['name']}: {obj['bbox']}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO object detection with Eigen-CAM visualization (wpipe)."
    )
    parser.add_argument(
        "--model", required=True, help="Path to YOLO weights file (.pt)"
    )
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument(
        "--output-dir",
        default="./output/detection",
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
    run_detection_example(
        args.model, args.image, args.output_dir, args.device, args.conf
    )
