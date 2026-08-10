"""Example: process a list of images with YOLO + ECAM visualization.

Usage:
    python example_batch_images.py --model yolov8n.pt --images bus.jpg zidane.jpg
                                   [--output-dir ./output/batch]
                                   [--device cpu] [--conf 0.25]
"""

import argparse
import os
from typing import Any

from loguru import logger
from wpipe import Pipeline

# Importing from the installed library
from wpipe_plugins.vision.ecam_yolo import ECAMConfig, ImageECamYOLO


def run_batch_images_example(
    model_path: str,
    image_paths: list[str],
    output_dir: str = "./output/batch",
    device: str = "cpu",
    confidence_threshold: float = 0.25,
) -> None:
    """Process a list of images, generating one ECAM visualization per image.

    Args:
        model_path (str): Path to the YOLO weights file (``.pt``).
        image_paths (list[str]): List of paths to the input images.
        output_dir (str): Directory where the visualizations are saved.
        device (str): Compute device, ``'cpu'`` or a GPU index like ``'0'``.
        confidence_threshold (float): Minimum confidence for a valid prediction.
    """
    # 1. Setup the Pipeline with an ECAM step
    pipe = Pipeline(pipeline_name="batch_images_example_pipeline", verbose=True)
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

    # 2. Prepare Input Data: the whole list goes under "image_data"
    os.makedirs(output_dir, exist_ok=True)

    inference_data = {
        "image_data": image_paths,
        "output_dir": output_dir,
    }

    # 3. Run the Pipeline
    logger.info(f"Starting batch pipeline for {len(image_paths)} images...")
    try:
        context: dict[str, Any] = pipe.run(inference_data)
        results: list[dict[str, Any]] = context.get("results", [])

        logger.info(f"Processed {len(results)} image(s)")
        for result in results:
            detections = [
                obj for obj in result["model_results"] if obj.get("status") == "ok"
            ]
            logger.info(f" - {len(detections)} detection(s)")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO + Eigen-CAM over a list of images (wpipe)."
    )
    parser.add_argument(
        "--model", required=True, help="Path to YOLO weights file (.pt)"
    )
    parser.add_argument(
        "--images", nargs="+", required=True, help="Paths to the input images"
    )
    parser.add_argument(
        "--output-dir",
        default="./output/batch",
        help="Directory to save the visualizations",
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
    run_batch_images_example(
        args.model, args.images, args.output_dir, args.device, args.conf
    )
