"""Pure data helpers used by the interactive image-processing views."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.operation_registry import ControlSpec, OperationSpec


def image_as_png(image: np.ndarray) -> bytes:
    """Encode an RGB or grayscale NumPy image as PNG bytes."""

    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def format_parameter_value(value: Any) -> str:
    """Format a control value for compact display in the interface."""

    if isinstance(value, tuple):
        return " – ".join(str(item) for item in value)
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _snap_to_control(value: float, control: ControlSpec) -> int | float:
    minimum = float(control.minimum)
    maximum = float(control.maximum)
    step = float(control.step or 1)
    snapped = minimum + round((value - minimum) / step) * step
    snapped = max(minimum, min(maximum, snapped))
    if isinstance(control.default, int):
        return round(snapped)
    return round(snapped, 4)


def sweep_variants(
    operation: OperationSpec,
    parameters: dict[str, Any],
) -> tuple[ControlSpec, list[tuple[str, dict[str, Any]]]] | None:
    """Build lower/current/higher parameter variants for an operation sweep."""

    if operation.sweep_parameter is None:
        return None

    control = next(
        item for item in operation.controls if item.key == operation.sweep_parameter
    )
    current = float(parameters[control.key])
    span = float(control.maximum) - float(control.minimum)
    offset = max(float(control.step or 1), span * 0.18)
    values = [
        _snap_to_control(current - offset, control),
        _snap_to_control(current, control),
        _snap_to_control(current + offset, control),
    ]

    unique_values: list[int | float] = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)

    variants: list[tuple[str, dict[str, Any]]] = []
    for value in unique_values:
        variant = parameters.copy()
        variant[control.key] = value
        variants.append((format_parameter_value(value), variant))
    return control, variants


def luminance_histogram(image: np.ndarray) -> np.ndarray:
    """Return a normalized 256-bin luminance histogram."""

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    counts, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = max(1, counts.sum())
    return counts / total
