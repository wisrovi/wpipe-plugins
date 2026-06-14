"""
Module Name
-----------
diagnost-ecam-yolo

Version
-------
1.0.0

Author
------
Ernesto Fernández

Date
----
2026-05-29

Description
-----------
This class performs image inference with integrated Eigen-CAM visualization.
It automatically supports both object detection and image classification workflows,
generating heatmaps that highlight the regions influencing the model predictions.
The pipeline handles image preprocessing, inference execution, visualization rendering,
and output generation with configurable settings and error handling.
"""

from dataclasses import dataclass
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch.nn as nn
from loguru import logger
from PIL import Image
from pytorch_grad_cam import EigenCAM
from torchvision import transforms
from ultralytics import YOLO
from wpipe import step, to_obj

# --- Internal Architecture Imports ---
from ..config.constants import DEVICE
from ..exceptions.yoloError import YoloError
from ..schemas.inference import InferenceObj, Path
from ..utils.vision_helpers import (
    parse_classification,
    parse_detections,
    parse_segmentation,
    process_classification_cam,
    process_segmentation_cam,
    renormalize_cam_in_bounding_boxes,
)
from ..wrappers.yolo_wrapper import YOLOUniversalWrapper


@dataclass
class ECAMConfig:
    """Configuration settings for Grad-CAM execution.

    Attributes:
        model_path (str): Path to the YOLO model weights.
        confidence_threshold (float): Minimum confidence to consider a valid prediction.
        device (Union[str, int]): Computation device to use (e.g., 'cpu', 0 for GPU).
    """

    model_path: str
    confidence_threshold: float = 0.25
    device: Union[str, int] = DEVICE.CPU


@step(name="ecam_yolo_inference", version="v1.0.0", timeout=30)
class ImageECamYOLO:
    """YOLO inference wrapper with integrated Eigen-CAM visualization.

    This class automatically detects whether the loaded YOLO model is a
    classification or detection model and applies the corresponding inference
    and Eigen-CAM processing pipeline.
    Attributes:
        config (Optional[ECAMConfig]): Configuration object containing model parameters.
        yolo_model (Optional[YOLO]): Loaded Ultralytics YOLO model instance.
        task (str): Indicates the specific task type of the model (e.g., 'detect', 'classify').
    """

    def __init__(self, config: Optional[ECAMConfig] = None) -> None:
        """Initialize the YOLO Grad-CAM inference wrapper.

        Args:
            config (Optional[ECAMConfig]): Configuration object containing
                model path, confidence threshold, and device settings. Defaults to None.

        Raises:
            YoloError: If the YOLO model cannot be loaded from the specified path.
        """
        self.config = config

        try:
            self.yolo_model: Optional[YOLO] = (
                YOLO(config.model_path) if config else None
            )

            # Automatically detect the YOLO task type.
            self.task = (
                getattr(self.yolo_model, "task", "detect")
                if self.yolo_model
                else "detect"
            )

            if self.config:
                logger.info(f"MODEL PATH: {self.config.model_path}")
                logger.info(f"MODEL TYPE: {type(self.yolo_model)}")

        except (FileNotFoundError, RuntimeError, OSError) as e:
            raise YoloError(f"Error loading model: {e}") from e

    def _get_target_layers(self) -> list[nn.Module]:
        """Retrieves the optimal feature map layers across all YOLO variants and tasks.

        This method dynamically inspects the loaded architecture to capture the most
        representative spatial feature maps. It adjusts automatically to separate
        needs: targeting the SPAN/Neck block for detection/segmentation tasks to
        preserve geometry, and extracting the final pooling/convolutional layers for
        classification models.

        Returns:
            List[nn.Module]: A list containing the selected PyTorch module node
                optimized for forward/backward activation mapping tracking.

        Raises:
            RuntimeError: If the internal structural audit cannot find a viable target.
        """
        try:
            if self.task == "classify":
                # Classification models require deep features. We target a layer slightly
                # before the absolute final pooling to preserve better spatial resolution.
                classify_layers = []
                for _, module in self.yolo_model.model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        classify_layers.append(module)

                if classify_layers:
                    # If the map looks too dotted/small, try changing index from [-1] to [-2] or [-3]
                    # [-3] or [-2] often catches the features prior to final spatial collapse.
                    selected_layer = (
                        classify_layers[-2]
                        if len(classify_layers) > 1
                        else classify_layers[-1]
                    )
                    logger.info(
                        f"Classification target layer adjusted: {type(selected_layer)}"
                    )
                    return [selected_layer]

            if hasattr(self.yolo_model.model, "model") and isinstance(
                self.yolo_model.model.model, nn.Sequential
            ):
                # Standard anchor-free detection/segmentation layouts bundle the Head at index [-1].
                # Index [-2] captures the combined multi-scale Neck (SPAN) output maps natively.
                target_module = self.yolo_model.model.model[-2]
                logger.info(
                    f"Multi-task target layer selected via Sequential SPAN index [-2]: {type(target_module)}"
                )
                return [target_module]

            # We filter out intermediate internal sub-convolutions responsible for bbox boundaries ("cv2", "cv3")
            # to prioritize clean, high-resolution visual feature maps.
            valid_convs = []
            for name, module in self.yolo_model.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    # Strip bounding-box regression layers from candidate pool
                    if not any(
                        head_keyword in name
                        for head_keyword in ["cv2", "cv3", "head", "reg"]
                    ):
                        valid_convs.append(module)

            if valid_convs:
                logger.info(
                    "Target layer selected via deep feature extraction Conv2d filtering."
                )
                return [valid_convs[-1]]

            raise RuntimeError(
                "No mathematically viable deep layers identified inside the architecture."
            )

        except (AttributeError, IndexError, TypeError) as e:
            logger.warning(
                f"Dynamic layer discovery auditing failed, executing emergency fallback: {e}"
            )
            # Global structural safety fallback (safest node layout for mainstream PyTorch wrappers)
            return [self.yolo_model.model.model[-2]]

    def _process_task(
        self, results: Any, img_float: np.ndarray, grayscale_cam: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Process YOLO outputs depending on the specific model task type.

        Args:
            results (Any): Raw output results from the YOLO model prediction.
            img_float (np.ndarray): Input image array normalized between 0 and 1.
            grayscale_cam (np.ndarray): Generated 2D Grad-CAM heatmap array.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: A tuple containing the processed
                visualization image array and a dictionary with detection results.
        """

        # CLASSIFICATION
        if self.task == "classify":

            if not results:
                return img_float, {"model_results": [{"status": "no_detections_found"}]}

            top1_idx, top1_name, top1_conf = parse_classification(results)

            final_img = process_classification_cam(
                img_float, grayscale_cam, top1_name, top1_conf
            )

            return final_img, {
                "model_results": [
                    {
                        "class_id": top1_idx,
                        "name": top1_name,
                        "confidence": top1_conf,
                        "status": "ok",
                    }
                ]
            }

        # SEGMENTATION
        if self.task == "segment":

            if results[0].masks is None or len(results[0].boxes) == 0:
                return img_float, {"model_results": [{"status": "no_detections_found"}]}

            boxes, masks_xy, colors, names = parse_segmentation(results)

            final_img = process_segmentation_cam(
                boxes, masks_xy, colors, names, img_float, grayscale_cam
            )

            return final_img, {
                "model_results": [
                    {
                        "class_id": i,
                        "name": n,
                        "bbox": b.tolist(),
                        "points_count": len(m),
                        "status": "ok",
                    }
                    for i, (b, m, n) in enumerate(zip(boxes, masks_xy, names))
                ]
            }

        # DETECTION (DEFAULT)
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return img_float, {"model_results": [{"status": "no_detections_found"}]}

        boxes, colors, names = parse_detections(results)

        final_img = renormalize_cam_in_bounding_boxes(
            boxes, colors, names, img_float, grayscale_cam
        )

        return final_img, {
            "model_results": [
                {
                    "class_id": i,
                    "name": n,
                    "bbox": b.tolist(),
                    "status": "ok",
                }
                for i, (b, n) in enumerate(zip(boxes, names))
            ]
        }

    def _save_visualization(
        self, final_img: np.ndarray, output_path: Path, output_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Save Grad-CAM / YOLO visualization and attach it to output results.

        Args:
            final_img (np.ndarray): Final generated image array containing visual results.
            output_path (Path): System destination path to save the generated image.
            output_results (Dict[str, Any]): Dictionary of model outputs to be updated.

        Returns:
            Dict[str, Any]: The updated dictionary incorporating the visualization data.
        """

        # 1. Convert float image [0,1] -> uint8
        final_img_uint8 = (final_img * 255).astype(np.uint8)

        # 2. Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Save image
        Image.fromarray(final_img_uint8).save(output_path)

        # 4. Attach visualization to output
        output_results["visualization"] = final_img_uint8

        return output_results

    @to_obj(InferenceObj)
    def __call__(self, image_info: InferenceObj) -> dict[str, Any]:
        """Run YOLO inference and generate Eigen-CAM visualizations.

        Depending on the detected YOLO task type, this method executes either
        the classification, segmentation, or detection Eigen-CAM workflow.

        Args:
            image_info (InferenceObj): Object containing image data, metadata,
                output configuration, and verbosity flags.

        Returns:
            Dict[str, Any]: Dictionary containing inference results and metadata.

        Raises:
            YoloError: If configuration is missing or inference fails.
        """
        if not self.config or not self.yolo_model:
            raise YoloError("Configuration or Model not found/loaded.")

        if getattr(image_info, "verbose", False) and self.yolo_model:
            logger.info(f"Model detected successfully. Task Type: {self.task.upper()}")

        try:

            if getattr(image_info, "verbose", False):
                if isinstance(image_info.image_data, np.ndarray):
                    logger.info(f"shape: {image_info.image_data.shape}")
                    logger.info(
                        f"min/max: {image_info.image_data.min()}/{image_info.image_data.max()}"
                    )
                else:
                    logger.info(
                        f"image_data type: {type(image_info.image_data).__name__}"
                    )
                    logger.info(f"image_data value: {image_info.image_data}")

            # Run YOLO inference.
            results = self.yolo_model.predict(
                source=image_info.image_data,
                conf=self.config.confidence_threshold,
                device=self.config.device,
                verbose=getattr(image_info, "verbose", False),
                imgsz=640,
            )

            if results is None:
                raise YoloError("Inference returned None results.")

            all_results = []
            for result in results:
                orig_bgr = result.orig_img
                orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                img_float = orig_rgb.astype(np.float32) / 255.0

                if getattr(image_info, "verbose", False):
                    r = result

                    logger.info(f"type(results): {type(result)}")
                    logger.info(f"boxes: {0 if r.boxes is None else len(r.boxes)}")
                    logger.info(f"masks: {None if r.masks is None else len(r.masks)}")

                # Configure Grad-CAM / EigenCAM.
                wrapped_model = YOLOUniversalWrapper(self.yolo_model.model)
                target_layers = self._get_target_layers()

                CAM_SIZE = 640
                orig_rgb_cam = cv2.resize(orig_rgb, (CAM_SIZE, CAM_SIZE))
                img_tensor = transforms.ToTensor()(orig_rgb_cam).unsqueeze(0)
                device_param = next(self.yolo_model.model.parameters()).device
                img_tensor = img_tensor.to(device_param)

                cam = EigenCAM(model=wrapped_model, target_layers=target_layers)
                grayscale_cam = cam(img_tensor)[0, :, :]
                h, w = img_float.shape[:2]
                grayscale_cam = cv2.resize(grayscale_cam, (w, h))

                if getattr(image_info, "verbose", False):
                    logger.info(f"CAM shape: {grayscale_cam.shape}")
                    logger.info(
                        f"CAM min/max: {grayscale_cam.min()} / {grayscale_cam.max()}"
                    )

                # Execute the workflow depending on the task type.
                final_img, output_results = self._process_task(
                    [result], img_float, grayscale_cam
                )

                result_path_str = result.path if result.path is not None else image_info.image_name
                output_path = Path(image_info.output_dir) / Path(result_path_str).name
                if (
                    getattr(image_info, "save", True)
                    or image_info.output_dir != "./output"
                ):
                    output_results = self._save_visualization(
                        final_img, output_path, output_results
                    )

                all_results.append(output_results)

            if getattr(image_info, "verbose", False) and all_results:
                logger.info(
                    "Inference completed successfully. Visualization generated in-memory."
                )
                logger.info(
                    f"Eigen-CAM visualization saved successfully at: {output_path}"
                )

            return all_results

        except (ValueError, RuntimeError, TypeError, AttributeError, YoloError) as e:
            logger.error(f"Inference failed: {e}")

            if isinstance(e, YoloError):
                raise e

            raise YoloError(str(e)) from e
