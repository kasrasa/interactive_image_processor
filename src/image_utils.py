"""Image loading, sizing, and sample-image utilities."""

from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps


def create_sample_image(width: int = 960, height: int = 640) -> np.ndarray:
    """Create a deterministic RGB image with useful edges, colors, and texture."""
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]

    red = np.broadcast_to(35 + 155 * x, (height, width))
    green = np.broadcast_to(45 + 115 * y, (height, width))
    blue = 155 + 55 * np.sin(2 * np.pi * (x[None, :] + 0.35 * y))
    image = np.stack((red, green, blue), axis=2).clip(0, 255).astype(np.uint8)

    overlay = image.copy()
    cv2.rectangle(overlay, (55, 55), (360, 280), (242, 184, 64), -1)
    cv2.circle(overlay, (500, 170), 112, (44, 201, 161), -1)
    cv2.line(overlay, (650, 70), (890, 285), (246, 246, 244), 18, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)

    checker_size = 32
    for row in range(6):
        for col in range(9):
            if (row + col) % 2 == 0:
                x0 = 58 + col * checker_size
                y0 = 360 + row * checker_size
                cv2.rectangle(
                    image,
                    (x0, y0),
                    (x0 + checker_size, y0 + checker_size),
                    (30, 39, 58),
                    -1,
                )

    cv2.putText(
        image,
        "OpenCV",
        (410, 445),
        cv2.FONT_HERSHEY_DUPLEX,
        2.25,
        (250, 250, 248),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "IMAGE PROCESSOR",
        (414, 505),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        (24, 34, 52),
        2,
        cv2.LINE_AA,
    )
    cv2.ellipse(image, (790, 470), (100, 62), -18, 0, 360, (224, 82, 109), -1)
    cv2.circle(image, (790, 470), 24, (252, 242, 232), -1)
    return image


def load_uploaded_image(uploaded_file: BinaryIO | BytesIO) -> np.ndarray:
    """Decode an uploaded image and return an RGB uint8 array."""
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)

    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")

    return np.asarray(image, dtype=np.uint8)


def convert_to_grayscale_rgb(image: np.ndarray) -> np.ndarray:
    """Return a grayscale rendering with three RGB-compatible channels.

    Keeping three identical channels lets every registered operation accept the
    same input shape while the pixels remain visually and numerically achromatic.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def is_grayscale_image(image: np.ndarray) -> bool:
    """Return whether an image is single-channel or has equal RGB channels."""
    if image.ndim == 2:
        return True
    if image.ndim != 3 or image.shape[2] != 3:
        return False
    return bool(
        np.array_equal(image[:, :, 0], image[:, :, 1])
        and np.array_equal(image[:, :, 1], image[:, :, 2])
    )


def resize_to_limit(image: np.ndarray, max_side: int = 1400) -> tuple[np.ndarray, bool]:
    """Downsize large inputs while preserving aspect ratio."""
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image, False

    scale = max_side / longest_side
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, True


def image_dimensions(image: np.ndarray) -> str:
    """Return dimensions formatted as width x height."""
    height, width = image.shape[:2]
    return f"{width} × {height}"


def mean_luminance(image: np.ndarray) -> float:
    """Return the average 8-bit grayscale luminance."""
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray))
