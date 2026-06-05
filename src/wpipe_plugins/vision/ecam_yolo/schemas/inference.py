"""Data Transfer Objects (DTOs) for inference data validation."""

import uuid
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class InferenceObj(BaseModel):
    """Data model for image inference input.

    Attributes:
        image_data (Union[str, Path, int, Image.Image, List[Any], Tuple[Any, ...], np.ndarray, torch.Tensor, None]):
            Input image data in various supported formats.
        image_name (str): Name of the image file (including extension).
        output_dir (str): Directory where the generated visualizations will be saved.
        reshape_transform (Optional[Callable]): Optional transform function for CAM spatial dimensions.
        verbose (bool): Whether to enable verbose logging during inference.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_data: (
        str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor
    ) = None
    image_name: str = Field(default_factory=lambda: f"{uuid.uuid4()}.png")
    save: bool = True
    output_dir: str = "./output"
    reshape_transform: Callable = None
    verbose: bool = False
