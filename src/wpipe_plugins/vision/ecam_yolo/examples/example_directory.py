"""Example: process every image inside a directory with ECAM visualization.

Usage:
    python example_directory.py --model yolov8n.pt --images-dir ./images
                                [--output-dir ./output/directory]
                                [--device cpu] [--conf 0.25]
"""

import argparse
import os
from typing import Any

from loguru import logger
from wpipe import Pipeline

# Importing from the installed library
from wpipe_plugins.vision.ecam_yolo import ECAMConfig, ImageECamYOLO


def run_directory_example(
    model_path: str,
    images_dir: str,
    output_dir: str = "./output/directory",
    device: str = "cpu",
    confidence_threshold: float = 0.25,
) -> None:
    """Process all images in a directory, one ECAM visualization per image.

    Args:
        model_path (str): Path to the YOLO weights file (``.pt``).
        images_dir (str): Directory containing the input images.
        output_dir (str): Directory where the visualizations are saved.
        device (str): Compute device, ``'cpu'`` or a GPU index like ``'0'``.
        confidence_threshold (float): Minimum confidence for a valid prediction.
    """
    # 1. Setup the Pipeline with an ECAM step
    pipe = Pipeline(pipeline_name="directory_example_pipeline", verbose=True)
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

    # 2. Prepare Input Data: the directory path goes under "image_data" and the
    #    step expands it to every image file inside it.
    os.makedirs(output_dir, exist_ok=True)

    inference_data = {
        "image_data": images_dir,
        "output_dir": output_dir,
    }

    # 3. Run the Pipeline
    logger.info(f"Starting directory pipeline over '{images_dir}'...")
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
        description="YOLO + Eigen-CAM over all images in a directory (wpipe)."
    )
    parser.add_argument(
        "--model", required=True, help="Path to YOLO weights file (.pt)"
    )
    parser.add_argument(
        "--images-dir", required=True, help="Directory with the input images"
    )
    parser.add_argument(
        "--output-dir",
        default="./output/directory",
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
    run_directory_example(
        args.model, args.images_dir, args.output_dir, args.device, args.conf
    )
