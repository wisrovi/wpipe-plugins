"""Example script for YOLO object detection with ECAM visualization."""

import os
from typing import Any, Dict, List

from loguru import logger
from wpipe import Pipeline

# Importing from the plugin
from ..states.eCam_yolo import ECAMConfig, ImageECamYOLO


def run_detection_example():
    """Demonstrates how to run object detection with ECAM."""
    
    # 1. Setup Configuration for Detection
    config = ECAMConfig(
        model_path="/models/yolo26l/yolo26l-detec.pt",
        confidence_threshold=0.3,
        device="cpu"
    )

    # 2. Initialize the ECAM Step
    detection_step = ImageECamYOLO(config)

    # 3. Setup the Pipeline
    pipe = Pipeline(
        pipeline_name="detection_example_pipeline",
        verbose=True
    )
    pipe.set_steps([detection_step])

    # 4. Prepare Input Data
    input_image = "/media/sample_image.jpg"
    output_dir = "./output/detection"
    
    os.makedirs(output_dir, exist_ok=True)

    inference_data = {
        "image_data": input_image,
        "image_name": "sample_detection.jpg",
        "save": True,
        "output_dir": output_dir,
        "verbose": True
    }

    # 5. Run the Pipeline
    logger.info("Starting detection pipeline...")
    try:
        results: List[Dict[str, Any]] = pipe.run(inference_data)
        
        for idx, result in enumerate(results):
            logger.info(f"Detected Objects: {len(result['model_results'])}")
            for obj in result['model_results']:
                if obj.get('status') == 'ok':
                    logger.info(f" - {obj['name']}: {obj['bbox']}")
                
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    run_detection_example()
