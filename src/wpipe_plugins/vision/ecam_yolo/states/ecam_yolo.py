"""YOLO image inference with integrated Eigen-CAM visualization.

This module performs image inference with integrated Eigen-CAM visualization.
It automatically supports object detection, image classification, and image
segmentation workflows, generating heatmaps that highlight the regions
influencing the model predictions. The pipeline handles image preprocessing,
inference execution, visualization rendering, and output generation with
configurable settings and error handling.

Module Name:
    diagnost-ecam-yolo

Version:
    1.0.0

Author:
    Ernesto Fernández

Date:
    2026-05-29
"""

import uuid
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from pytorch_grad_cam import EigenCAM
from torch import nn
from torchvision import transforms
from ultralytics import YOLO
from wpipe import step, to_obj

# --- Internal Architecture Imports ---
from ..config.constants import DEVICE
from ..exceptions.yolo_error import YoloError
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

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class ECAMConfig:
    """Configuration settings for Grad-CAM execution.

    Attributes:
        model_path (str): Path to the YOLO model weights. Optional when the model
            is provided at run time (``model_path`` in the pipeline input data).
        confidence_threshold (float): Minimum confidence to consider a valid
            prediction. Defaults to 0.25.
        device (str | int | None): Compute device to use (e.g., 'cpu', 0 for GPU).
            When None (default) it is auto-detected at run time: GPU is used when
            available, otherwise CPU. An explicit value is used as-is without
            checking GPU availability.
    """

    model_path: str = ""
    confidence_threshold: float = 0.25
    device: str | int | None = None


@step(name="ecam_yolo_inference", version="v1.0.0", timeout=30)
class ImageECamYOLO:
    """YOLO inference wrapper with integrated Eigen-CAM visualization.

    This class automatically detects whether the loaded YOLO model is a
    classification or detection model and applies the corresponding inference
    and Eigen-CAM processing pipeline.

    Attributes:
        config (ECAMConfig | None): Configuration object containing model parameters.
        yolo_model (YOLO | None): Loaded Ultralytics YOLO model instance.
        task (str): Indicates the specific task type of the model (e.g., 'detect', 'classify').
    """

    def __init__(self, config: ECAMConfig | None = None) -> None:
        """Initialize the YOLO Grad-CAM inference wrapper.

        The model is loaded eagerly only when ``config.model_path`` is provided.
        When the pipeline runs, the ``model_path`` key of the input data can
        override (or provide, if missing) the model used for that inference.

        Args:
            config (ECAMConfig | None): Configuration object containing
                model path, confidence threshold, and device settings. Defaults to None.

        Raises:
            YoloError: If the YOLO model cannot be loaded from the specified path.
        """
        self.config = config or ECAMConfig()

        if self.config.model_path:
            self._load_model(self.config.model_path)
        else:
            self.yolo_model: YOLO | None = None
            self.task: str = "detect"

    def _load_model(self, model_path: str) -> None:
        """Load a YOLO model and detect its task type.

        Args:
            model_path (str): Path to the YOLO model weights.

        Raises:
            YoloError: If the YOLO model cannot be loaded from the specified path.
        """
        try:
            self.yolo_model = YOLO(model_path)

            # Automatically detect the YOLO task type.
            self.task = getattr(self.yolo_model, "task", "detect")
            self.config.model_path = model_path
            logger.info(f"MODEL PATH: {model_path}")
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
                # Strip bounding-box regression layers from candidate pool
                if isinstance(module, nn.Conv2d) and not any(
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
                    for i, (b, m, n) in enumerate(zip(boxes, masks_xy, names, strict=True))
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
                for i, (b, n) in enumerate(zip(boxes, names, strict=True))
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

    def _compute_grayscale_cam(self, orig_rgb: np.ndarray) -> np.ndarray:
        """Compute the Eigen-CAM heatmap for an RGB image (uint8).

        Args:
            orig_rgb (np.ndarray): Original image in RGB (uint8, HxWx3).

        Returns:
            np.ndarray: Grayscale Eigen-CAM heatmap at the original image size.
        """
        wrapped_model = YOLOUniversalWrapper(self.yolo_model.model)
        target_layers = self._get_target_layers()

        cam_size = 640
        orig_rgb_cam = cv2.resize(orig_rgb, (cam_size, cam_size))
        img_tensor = transforms.ToTensor()(orig_rgb_cam).unsqueeze(0)
        device_param = next(self.yolo_model.model.parameters()).device
        img_tensor = img_tensor.to(device_param)

        cam = EigenCAM(model=wrapped_model, target_layers=target_layers)
        h, w = orig_rgb.shape[:2]
        return cv2.resize(cam(img_tensor)[0, :, :], (w, h))

    def _process_result(
        self, result: Any, image_info: InferenceObj, verbose: bool
    ) -> tuple[dict[str, Any], Path]:
        """Run inference and CAM for a single YOLO result, saving the visualization.

        Args:
            result (Any): A single Ultralytics inference result.
            image_info (InferenceObj): Input data for the run.
            verbose (bool): Whether to enable verbose logging.

        Returns:
            Tuple[Dict[str, Any], Path]: The output results and the saved path.
        """
        orig_bgr = result.orig_img
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        img_float = orig_rgb.astype(np.float32) / 255.0

        if verbose:
            logger.info(f"type(results): {type(result)}")
            logger.info(f"boxes: {0 if result.boxes is None else len(result.boxes)}")
            logger.info(f"masks: {None if result.masks is None else len(result.masks)}")

        grayscale_cam = self._compute_grayscale_cam(orig_rgb)

        if verbose:
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

        return output_results, output_path

    def _resolve_device(self) -> str | int:
        """Resolve the compute device to use for inference.

        When ``config.device`` is None the device is auto-detected: CUDA/GPU is
        used if available, otherwise CPU. An explicit config value is returned
        as-is without checking GPU availability.

        Returns:
            str | int: The compute device ('cpu', a CUDA index, 'cuda:0', ...).
        """
        if self.config.device is not None:
            return self.config.device
        return 0 if torch.cuda.is_available() else DEVICE.CPU

    def _resolve_image_name(self, source: Any) -> str:
        """Derive the visualization output name from a single image source.

        Args:
            source (Any): A single image source (path, PIL image, array, ...).

        Returns:
            str: The base file name for the output visualization.
        """
        if isinstance(source, (str, Path)):
            return Path(source).name
        return f"{uuid.uuid4()}.png"

    def _expand_sources(self, image_data: Any) -> list[Any]:
        """Normalize ``image_data`` into a flat list of individual image sources.

        A list/tuple is expanded item by item, a directory path is expanded to
        the sorted list of its image files, and any other single source is
        wrapped in a list. This lets batch, directory, and single-image inputs
        run through the same per-image processing loop.

        Args:
            image_data (Any): The raw ``image_data`` field from the run.

        Returns:
            List[Any]: A flat list of image sources (paths, arrays, PIL images...).
        """

        def _expand(item: Any) -> list[Any]:
            if item is None:
                return []
            if isinstance(item, (str, Path)) and Path(item).is_dir():
                return sorted(
                    p
                    for p in Path(item).iterdir()
                    if p.suffix.lower() in _IMAGE_EXTENSIONS
                )
            return [item]

        if isinstance(image_data, (list, tuple)):
            return [source for item in image_data for source in _expand(item)]
        return _expand(image_data)

    @to_obj(InferenceObj)
    def __call__(self, image_info: InferenceObj) -> dict[str, Any]:
        """Run YOLO inference and generate Eigen-CAM visualizations.

        Depending on the detected YOLO task type, this method executes either
        the classification, segmentation, or detection Eigen-CAM workflow.

        Args:
            image_info (InferenceObj): Object containing the image data and
                output configuration. ``image_data`` may be a single image, a
                directory of images, or a list/tuple of either; each image is
                processed and produces its own result visualization. The
                ``model_path`` key may select (or override) the model used for
                this run. The ``save``, ``verbose`` and ``image_name`` fields
                are set internally by this step and are not part of the run
                data contract.

        Returns:
            Dict[str, Any]: Dictionary containing inference results and metadata.
                When no model is available (neither in the config nor in the run
                data) it returns a graceful ``no_model`` result instead of failing.

        Raises:
            YoloError: If the model path cannot be loaded or inference fails.
        """
        # Fixed, immutable runtime settings: this step always saves the
        # Eigen-CAM visualization and runs quietly, so callers must not
        # provide them in the run data. ``image_name`` is derived per source.
        image_info.save = True
        image_info.verbose = False

        # Model selection at run time: the input data may provide (or override)
        # the model path used for this inference.
        run_model_path = getattr(image_info, "model_path", None)
        if run_model_path and run_model_path != self.config.model_path:
            logger.info(f"Loading model provided at run time: {run_model_path}")
            self._load_model(run_model_path)

        # Graceful degradation: without a model the step reports the reason and
        # returns a result, so the pipeline keeps running instead of failing.
        if not self.yolo_model:
            message = (
                "Inference skipped: no model available. Provide 'model_path' in "
                "the ECAMConfig or in the pipeline run data."
            )
            logger.error(message)
            return {
                "results": [
                    {"model_results": [{"status": "no_model", "message": message}]}
                ]
            }

        verbose = image_info.verbose
        if verbose:
            logger.info(f"Model detected successfully. Task Type: {self.task.upper()}")

        # Input normalization: a list, a directory, or a single image all become
        # a flat list of sources, each producing its own result visualization.
        sources = self._expand_sources(image_info.image_data)
        if not sources:
            message = (
                "Inference skipped: no images found for the provided 'image_data'."
            )
            logger.error(message)
            return {
                "results": [
                    {"model_results": [{"status": "no_images", "message": message}]}
                ]
            }

        try:
            return {"results": self._run_sources(sources, image_info, verbose)}

        except (ValueError, RuntimeError, TypeError, AttributeError, YoloError) as e:
            logger.error(f"Inference failed: {e}")

            if isinstance(e, YoloError):
                raise e

            raise YoloError(str(e)) from e

    def _run_sources(
        self, sources: list[Any], image_info: InferenceObj, verbose: bool
    ) -> list[dict[str, Any]]:
        """Run inference and CAM over each source, collecting the output results.

        Args:
            sources (List[Any]): Flat list of image sources to process.
            image_info (InferenceObj): Input data for the run.
            verbose (bool): Whether to enable verbose logging.

        Returns:
            List[Dict[str, Any]]: One output result dict per processed image.
        """
        all_results: list[dict[str, Any]] = []
        device = self._resolve_device()
        for source in sources:
            image_info.image_name = self._resolve_image_name(source)
            if verbose:
                logger.info(f"Processing source: {image_info.image_name}")
                if isinstance(source, np.ndarray):
                    logger.info(f"shape: {source.shape}")
                    logger.info(f"min/max: {source.min()}/{source.max()}")
                else:
                    logger.info(f"image_data type: {type(source).__name__}")
                    logger.info(f"image_data value: {source}")

            # Run YOLO inference for this source.
            results = self.yolo_model.predict(
                source=source,
                conf=self.config.confidence_threshold,
                device=device,
                verbose=verbose,
                imgsz=640,
            )

            if results is None:
                raise YoloError("Inference returned None results.")

            last_output_path: Path | None = None
            for result in results:
                output_results, output_path = self._process_result(
                    result, image_info, verbose
                )
                all_results.append(output_results)
                last_output_path = output_path

            if verbose and all_results:
                logger.info(
                    "Inference completed successfully. Visualization generated in-memory."
                )
                logger.info(
                    f"Eigen-CAM visualization saved successfully at: {last_output_path}"
                )

        return all_results
