"""Custom exceptions for the vision diagnostics and inference pipeline."""


class VisionPipelineError(Exception):
    """Base exception for all errors raised within the vision pipeline.

    Attributes:
        message (str): Explanation of the error.
        error_code (str | None): Standardized alphanumeric code for debugging.
    """

    def __init__(self, message: str, error_code: str | None = None) -> None:
        """Initialize the base vision pipeline exception.

        Args:
            message (str): Detailed description of the error.
            error_code (str | None): Optional error identifier code. Defaults to None.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def __str__(self) -> str:
        """Return the string representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class YoloError(VisionPipelineError):
    """Custom exception raised for errors during YOLO/Grad-CAM inference execution.

    This covers model loading failures, processing mismatches, or mathematical
    inconsistencies during backward/forward tracking layers.
    """

    def __init__(self, message: str, error_code: str = "YOLO_INFERENCE_ERROR") -> None:
        """Initialize the YOLO-specific inference exception."""
        super().__init__(message, error_code=error_code)
