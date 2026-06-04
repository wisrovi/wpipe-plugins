"""Universal wrapper for YOLO models to ensure standardized tensor outputs for Grad-CAM."""

from typing import Any, Optional

import torch


class YOLOUniversalWrapper(torch.nn.Module):
    """Wraps a YOLO model to guarantee a single flat tensor output.
    
    This is required for compatibility with PyTorch-Grad-CAM workflows, 
    as YOLO variants often return complex nested structures (tuples, dicts, lists) 
    containing raw predictions, anchors, and metadata.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Initialize the wrapper with a target model instance."""
        super().__init__()
        self.model = model

    def _find_first_tensor(self, obj: Any) -> Optional[torch.Tensor]:
        """Recursively search through nested structures to extract the primary output tensor.

        Args:
            obj (Any): The nested object structure to inspect (Tensor, list, tuple, or dict).

        Returns:
            Optional[torch.Tensor]: The first encountered PyTorch tensor, or None if not found.
        """
        if isinstance(obj, torch.Tensor):
            return obj

        if isinstance(obj, (list, tuple)):
            for item in obj:
                tensor = self._find_first_tensor(item)
                if tensor is not None:
                    return tensor

        if isinstance(obj, dict):
            for value in obj.values():
                tensor = self._find_first_tensor(value)
                if tensor is not None:
                    return tensor

        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute model forward pass and standardize the resulting output structure.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: A batch-like standardized output tensor ready for Grad-CAM.

        Raises:
            TypeError: If no valid tensor can be extracted from the model's raw outputs.
        """
        outputs = self.model(x)
        tensor = self._find_first_tensor(outputs)

        if tensor is None:
            raise TypeError(
                f"Could not extract a valid tensor from output structure type: {type(outputs)}"
            )

        # Grad-CAM requires a batch dimension layout (B, C, H, W) or (B, N, C)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        return tensor