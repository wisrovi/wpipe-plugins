"""Helper functions for image processing, drawing, and Grad-CAM blending."""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics.engine.results import Results

# Set seed for reproducible color generation
np.random.seed(42)
# Generate colors directly in np.uint8 to avoid compatibility issues with OpenCV
COLORS: np.ndarray = np.random.uniform(0, 255, size=(80, 3)).astype(np.uint8)


def parse_detections(
    results: List[Results],
) -> Tuple[List[np.ndarray], List[List[int]], List[str]]:
    """Extract detection data from an Ultralytics Results object.

    Filters detections by confidence score and extracts bounding box
    coordinates, class colors, and class names.

    Args:
        results (List[Results]): Ultralytics inference results object containing
            detection outputs.

    Returns:
        Tuple[List[np.ndarray], List[List[int]], List[str]]: A tuple containing:
            boxes (List[np.ndarray]): Bounding box coordinates in ``[x1, y1, x2, y2]`` format.
            colors (List[List[int]]): RGB color values associated with each class, e.g., ``[[255, 0, 0], ...]``.
            names (List[str]): Class names corresponding to each detection.
    """
    boxes: List[np.ndarray] = []
    colors: List[List[int]] = []
    names: List[str] = []

    res = results[0]

    if res.boxes:
        for box in res.boxes:
            conf = float(box.conf[0])
            if conf < 0.2:
                continue
            coords = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])

            boxes.append(coords)
            colors.append(COLORS[cls].tolist())
            names.append(res.names[cls])

    return boxes, colors, names


def draw_detections(
    boxes: List[np.ndarray],
    colors: List[List[int]],
    names: List[str],
    img: np.ndarray,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Note:
        OpenCV natively expects colors in BGR format for its drawing functions.
        If the provided ``colors`` list is in RGB format, the rendered boxes 
        will have their Red and Blue channels visually swapped.

    Args:
        boxes (List[np.ndarray]): List of bounding boxes in ``[x1, y1, x2, y2]`` format.
        colors (List[List[int]]): List of color values for each detection.
        names (List[str]): List of class names for each detection.
        img (np.ndarray): Input image where detections will be drawn.

    Returns:
        np.ndarray: Image with rendered bounding boxes and labels.
    """
    for box, color, name in zip(boxes, colors, names):
        x1, y1, x2, y2 = box
        # OpenCV uses native tuples or lists of ints for colors
        color_tuple = tuple(int(c) for c in color)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color_tuple, 2)
        cv2.putText(
            img,
            name,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_tuple,
            2,
            lineType=cv2.LINE_AA,
        )
    return img


def renormalize_cam_in_bounding_boxes(
    boxes: List[np.ndarray],
    colors: List[List[int]],
    names: List[str],
    image_float_np: np.ndarray,
    grayscale_cam: np.ndarray,
) -> np.ndarray:
    """Apply and normalize a Grad-CAM heatmap inside detected bounding boxes.

    The heatmap is normalized independently for each detected region and then
    blended with the original image to create a visualization focused on the
    detected objects.

    Args:
        boxes (List[np.ndarray]): List of bounding boxes in ``[x1, y1, x2, y2]`` format.
        colors (List[List[int]]): List of color values for each detection.
        names (List[str]): List of class names for each detection.
        image_float_np (np.ndarray): Original image as a normalized float array [0, 1].
        grayscale_cam (np.ndarray): Grayscale Grad-CAM heatmap.

    Returns:
        np.ndarray: RGB image containing the blended heatmap and rendered detections.
    """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)

    for x1, y1, x2, y2 in boxes:
        if y2 > y1 and x2 > x1:
            region = grayscale_cam[y1:y2, x1:x2]

            if region.max() > 0:
                renormalized_cam[y1:y2, x1:x2] = (region - region.min()) / (
                    region.max() - region.min() + 1e-8
                )

    renormalized_cam = cv2.GaussianBlur(renormalized_cam, (11, 11), 0)
    # Use np.round to prevent precision loss before converting to uint8
    heatmap_uint8 = np.round(renormalized_cam * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_float = heatmap_rgb.astype(np.float32) / 255.0

    cam_image = (image_float_np * 0.5) + (heatmap_float * 0.5)
    cam_image = np.clip(cam_image, 0, 1)

    return draw_detections(boxes, colors, names, cam_image)


def parse_classification(
    results: List[Results],
) -> Tuple[Optional[int], str, float]:
    """Extract top-1 classification data from an Ultralytics Results object.

    Args:
        results (List[Results]): Ultralytics inference results object containing
            classification outputs.

    Returns:
        Tuple[Optional[int], str, float]: A tuple containing:
            top1_idx (Optional[int]): Index of the predicted top-1 class (or None if unavailable).
            top1_name (str): Name of the predicted class.
            top1_conf (float): Confidence score of the predicted class.
    """
    res = results[0]

    if res.probs is not None:
        top1_idx = int(res.probs.top1)
        top1_name = str(res.names[top1_idx])
        top1_conf = float(res.probs.top1conf)

        return top1_idx, top1_name, top1_conf

    return None, "Unknown", 0.0


def process_classification_cam(
    image_float_np: np.ndarray,
    grayscale_cam: np.ndarray,
    class_name: str,
    confidence: float,
) -> np.ndarray:
    """Blends CAM to the full image using standard thermal (JET) colors.

    Args:
        image_float_np (np.ndarray): Original image as float32 [0, 1].
        grayscale_cam (np.ndarray): Raw CAM mask from EigenCAM.
        class_name (str): Name of the predicted class.
        confidence (float): Confidence score of the prediction.

    Returns:
        np.ndarray: Blended image with heatmap and text overlay.
    """
    # Use np.round to prevent loss of thermal gradient precision
    grayscale_cam_uint8 = np.round(255 * grayscale_cam).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(grayscale_cam_uint8, cv2.COLORMAP_JET)

    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    heatmap_float = np.float32(heatmap_rgb) / 255.0

    cam_image = (0.5 * heatmap_float) + (0.5 * image_float_np)
    cam_image = np.clip(cam_image, 0, 1)

    cam_image_uint8 = np.round(255 * cam_image).astype(np.uint8)
    text = f"{class_name} ({confidence * 100:.1f}%)"
    cv2.putText(
        cam_image_uint8,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
        lineType=cv2.LINE_AA,
    )

    return np.float32(cam_image_uint8) / 255.0


def parse_segmentation(
    results: List[Results],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[int]], List[str]]:
    """Extract segmentation masks and bounding box data from Ultralytics Results.

    Args:
        results (List[Results]): Ultralytics inference results object.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[List[int]], List[str]]: A tuple containing:
            boxes (List[np.ndarray]): Bounding box coordinates in ``[x1, y1, x2, y2]`` format.
            masks_xy (List[np.ndarray]): Array of polygon points representing the mask outlines.
            colors (List[List[int]]): RGB color values associated with each class.
            names (List[str]): Class names corresponding to each detection.
    """
    boxes: List[np.ndarray] = []
    masks_xy: List[np.ndarray] = []
    colors: List[List[int]] = []
    names: List[str] = []

    res = results[0]
    if res.masks is not None and res.boxes is not None:
        for box, mask_xy in zip(res.boxes, res.masks.xy):
            conf = float(box.conf[0])
            if conf < 0.2:
                continue
            coords = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])

            boxes.append(coords)
            masks_xy.append(mask_xy.astype(np.int32))
            colors.append(COLORS[cls].tolist())
            names.append(res.names[cls])

    return boxes, masks_xy, colors, names


def process_segmentation_cam(
    boxes: List[np.ndarray],
    masks_xy: List[np.ndarray],
    colors: List[List[int]],
    names: List[str],
    image_float_np: np.ndarray,
    grayscale_cam: np.ndarray,
) -> np.ndarray:
    """Blends full-screen CAM heatmap overlayed with translucent segmentation masks.

    Args:
        boxes (List[np.ndarray]): List of bounding boxes.
        masks_xy (List[np.ndarray]): List of mask polygon coordinate arrays.
        colors (List[List[int]]): List of color values for each class.
        names (List[str]): List of class names.
        image_float_np (np.ndarray): Original image as float32 array [0, 1].
        grayscale_cam (np.ndarray): Grayscale Grad-CAM heatmap.

    Returns:
        np.ndarray: Blended image with heatmap, translucent masks, and bounding boxes.
    """
    # 1. Standard global CAM blending
    grayscale_cam_uint8 = np.round(255 * grayscale_cam).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(grayscale_cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    heatmap_float = heatmap_rgb.astype(np.float32) / 255.0

    cam_image = (image_float_np * 0.5) + (heatmap_float * 0.5)
    cam_image = np.clip(cam_image, 0, 1)
    cam_image_uint8 = np.round(255 * cam_image).astype(np.uint8)

    # 2. Render translucent masks
    mask_overlay = cam_image_uint8.copy()
    for mask, color in zip(masks_xy, colors):
        if len(mask) > 0:
            color_tuple = tuple(int(c) for c in color)
            cv2.fillPoly(mask_overlay, [mask], color_tuple)

    # Blend overlay with 30% mask transparency
    cv2.addWeighted(mask_overlay, 0.3, cam_image_uint8, 0.7, 0, cam_image_uint8)

    # 3. Draw bounding boxes and text labels on top
    final_img_float = cam_image_uint8.astype(np.float32) / 255.0
    return draw_detections(boxes, colors, names, final_img_float)