"""Example script for YOLO image classification with ECAM visualization.

Usage:
    python example_classification.py --model yolov8n-cls.pt --image cat.jpg
                                     [--output-dir ./output/classification]
                                     [--device cpu] [--conf 0.25]
"""

import argparse
import os
from typing import Any

from loguru import logger
from wpipe import Pipeline

# Importing from the installed library
from wpipe_plugins.vision.ecam_yolo import ECAMConfig, ImageECamYOLO


def run_classification_example(
    model_path: str,
    image_path: str,
    output_dir: str = "./output/classification",
    device: str = "cpu",
    confidence_threshold: float = 0.25,
) -> None:
    """Demonstrates how to run classification with ECAM.

    Args:
        model_path (str): Path to the YOLO classification weights file (``.pt``).
        image_path (str): Path to the input image.
        output_dir (str): Directory where the visualization is saved.
        device (str): Compute device, ``'cpu'`` or a GPU index like ``'0'``.
        confidence_threshold (float): Minimum confidence for a valid prediction.
    """
    pipe = Pipeline(pipeline_name="classification_example_pipeline", verbose=True)
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
    logger.info("Starting classification pipeline...")
    try:
        context: dict[str, Any] = pipe.run(inference_data)
        results: list[dict[str, Any]] = context.get("results", [])

        for idx, result in enumerate(results):
            logger.info(f"Result {idx}: {result['model_results']}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO image classification with Eigen-CAM visualization (wpipe)."
    )
    parser.add_argument(
        "--model", required=True, help="Path to YOLO classification weights file (.pt)"
    )
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument(
        "--output-dir",
        default="./output/classification",
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
    run_classification_example(
        args.model, args.image, args.output_dir, args.device, args.conf
    )
