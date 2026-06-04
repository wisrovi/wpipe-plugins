"""Example script for YOLO image segmentation with ECAM visualization."""

import os
from typing import Any, Dict, List

from loguru import logger
from wpipe import Pipeline

# Importing from the plugin
from ..states.eCam_yolo import ECAMConfig, ImageECamYOLO


def run_segmentation_example():
    """Demonstrates how to run image segmentation with ECAM."""
    
    # 1. Setup Configuration for Segmentation
    config = ECAMConfig(
        model_path="/models/yolo26l/yolo26l-seg.pt",
        confidence_threshold=0.3,
        device="cpu"
    )

    # 2. Initialize the ECAM Step
    segmentation_step = ImageECamYOLO(config)

    # 3. Setup the Pipeline
    pipe = Pipeline(
        pipeline_name="segmentation_example_pipeline",
        verbose=True
    )
    pipe.set_steps([segmentation_step])

    # 4. Prepare Input Data
    input_image = "/media/sample_image.jpg"
    output_dir = "./output/segmentation"
    
    os.makedirs(output_dir, exist_ok=True)

    inference_data = {
        "image_data": input_image,
        "image_name": "sample_segmentation.jpg",
        "save": True,
        "output_dir": output_dir,
        "verbose": True
    }

    # 5. Run the Pipeline
    logger.info("Starting segmentation pipeline...")
    try:
        results: List[Dict[str, Any]] = pipe.run(inference_data)
        
        for result in enumerate(results):
            logger.info(f"Segmented Objects: {len(result['model_results'])}")
            for obj in result['model_results']:
                if obj.get('status') == 'ok':
                    logger.info(f" - {obj['name']}: Bbox {obj['bbox']}, Points {obj['points_count']}")
                
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    run_segmentation_example()
