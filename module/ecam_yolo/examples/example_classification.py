"""Example script for YOLO image classification with ECAM visualization."""

import os
from typing import Any, Dict, List

from loguru import logger
from wpipe import Pipeline

# Importing from the plugin
from ..states.eCam_yolo import ECAMConfig, ImageECamYOLO


def run_classification_example():
    """Demonstrates how to run classification with ECAM."""
    
    # 1. Setup Configuration
    config = ECAMConfig(
        model_path="/models/yolo26l/yolo26l-cls.pt",
        confidence_threshold=0.3,
        device="cpu"
    )

    # 2. Initialize the ECAM Step
    classification_step = ImageECamYOLO(config)

    # 3. Setup the Pipeline
    pipe = Pipeline(
        pipeline_name="classification_example_pipeline",
        verbose=True
    )
    pipe.set_steps([classification_step])

    # 4. Prepare Input Data
    input_image = "/media/sample_image.jpg"
    output_dir = "./output/classification"
    
    os.makedirs(output_dir, exist_ok=True)

    inference_data = {
        "image_data": input_image,
        "image_name": "sample_classification.jpg",
        "save": True,
        "output_dir": output_dir,
        "verbose": True
    }

    # 5. Run the Pipeline
    logger.info("Starting classification pipeline...")
    try:
        results: List[Dict[str, Any]] = pipe.run(inference_data)
        
        for idx, result in enumerate(results):
            logger.info(f"Result {idx}: {result['model_results']}")
            if "output_path" in result:
                logger.info(f"Visualization saved at: {result['output_path']}")
                
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    run_classification_example()
