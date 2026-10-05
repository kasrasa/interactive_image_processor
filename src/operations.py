"""Pure OpenCV operations used by the interactive explorer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class OperationResult:
    """The rendered image plus small, operation-specific measurements."""

    image: np.ndarray
    metrics: dict[str, str] = field(default_factory=dict)


def _uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _normalize_for_display(values: np.ndarray, log_scale: bool) -> np.ndarray:
    """Map floating-point transform values to an 8-bit display image."""

    display_values = np.abs(values).astype(np.float32)
    if log_scale:
        display_values = np.log1p(display_values)
    return cv2.normalize(
        display_values,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)


def adjust_hsv(
    image: np.ndarray,
    hue_shift: int,
    saturation_scale: float,
    value_scale: float,
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = np.mod(hsv[:, :, 0] + hue_shift / 2.0, 180)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def adjust_lab(
    image: np.ndarray,
    lightness_scale: float,
    chroma_scale: float,
    a_shift: int,
    b_shift: int,
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * lightness_scale, 0, 255)
    lab[:, :, 1] = np.clip(
        (lab[:, :, 1] - 128.0) * chroma_scale + 128.0 + a_shift,
        0,
        255,
    )
    lab[:, :, 2] = np.clip(
        (lab[:, :, 2] - 128.0) * chroma_scale + 128.0 + b_shift,
        0,
        255,
    )
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def box_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    return cv2.blur(image, (kernel_size, kernel_size))


def gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)


def bilateral_filter(
    image: np.ndarray,
    diameter: int,
    sigma_color: float,
    sigma_space: float,
) -> np.ndarray:
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def fourier_transform(
    image: np.ndarray,
    view: str,
    center_frequency: bool,
    log_scale: bool,
) -> OperationResult:
    """Return a displayable magnitude or phase view of the 2D Fourier transform."""

    gray = _gray(image).astype(np.float32)
    spectrum = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    if center_frequency:
        spectrum = np.fft.fftshift(spectrum, axes=(0, 1))

    if view == "Phase spectrum":
        phase = cv2.phase(spectrum[:, :, 0], spectrum[:, :, 1])
        output = np.clip(phase * (255.0 / (2.0 * np.pi)), 0, 255).astype(
            np.uint8
        )
    else:
        magnitude = cv2.magnitude(spectrum[:, :, 0], spectrum[:, :, 1])
        output = _normalize_for_display(magnitude, log_scale)

    zero_frequency = "Center" if center_frequency else "Top-left corner"
    return OperationResult(output, {"Zero frequency": zero_frequency})


def dct_transform(
    image: np.ndarray,
    view: str,
    low_frequency_span: int,
    log_scale: bool,
) -> OperationResult:
    """Visualize 2D DCT coefficients or reconstruct from a low-frequency block."""

    gray = _gray(image).astype(np.float32)
    height, width = gray.shape
    bottom_padding = height % 2
    right_padding = width % 2
    border_mode = (
        cv2.BORDER_REFLECT_101 if height > 1 and width > 1 else cv2.BORDER_REPLICATE
    )
    padded = cv2.copyMakeBorder(
        gray,
        0,
        bottom_padding,
        0,
        right_padding,
        border_mode,
    )
    coefficients = cv2.dct(padded)

    fraction = low_frequency_span / 100.0
    kept_height = max(1, int(np.ceil(coefficients.shape[0] * fraction)))
    kept_width = max(1, int(np.ceil(coefficients.shape[1] * fraction)))
    retained = np.zeros_like(coefficients)
    retained[:kept_height, :kept_width] = coefficients[
        :kept_height,
        :kept_width,
    ]
    kept_ratio = 100.0 * kept_height * kept_width / retained.size

    if view == "Low-frequency reconstruction":
        reconstruction = cv2.idct(retained)[:height, :width]
        output = np.clip(np.rint(reconstruction), 0, 255).astype(np.uint8)
    else:
        output = _normalize_for_display(retained, log_scale)

    return OperationResult(output, {"Coefficient block": f"{kept_ratio:.1f}%"})


def haar_wavelet_transform(
    image: np.ndarray,
    levels: int,
    detail_gain: float,
    log_scale: bool,
) -> OperationResult:
    """Build a multi-level 2D Haar coefficient mosaic without extra dependencies."""

    gray = _gray(image).astype(np.float32)
    height, width = gray.shape
    factor = 2**levels
    bottom_padding = (-height) % factor
    right_padding = (-width) % factor
    border_mode = (
        cv2.BORDER_REFLECT_101 if height > 1 and width > 1 else cv2.BORDER_REPLICATE
    )
    coefficients = cv2.copyMakeBorder(
        gray,
        0,
        bottom_padding,
        0,
        right_padding,
        border_mode,
    )

    current_height, current_width = coefficients.shape
    root_two = np.sqrt(2.0)
    for _ in range(levels):
        region = coefficients[:current_height, :current_width].copy()
        low_columns = (region[:, 0::2] + region[:, 1::2]) / root_two
        high_columns = (region[:, 0::2] - region[:, 1::2]) / root_two

        low_low = (low_columns[0::2] + low_columns[1::2]) / root_two
        high_low = (low_columns[0::2] - low_columns[1::2]) / root_two
        low_high = (high_columns[0::2] + high_columns[1::2]) / root_two
        high_high = (high_columns[0::2] - high_columns[1::2]) / root_two

        half_height = current_height // 2
        half_width = current_width // 2
        coefficients[:half_height, :half_width] = low_low
        coefficients[:half_height, half_width:current_width] = low_high
        coefficients[half_height:current_height, :half_width] = high_low
        coefficients[
            half_height:current_height,
            half_width:current_width,
        ] = high_high
        current_height, current_width = half_height, half_width

    display_values = np.abs(coefficients)
    detail_mask = np.ones_like(display_values, dtype=bool)
    detail_mask[:current_height, :current_width] = False
    display_values[detail_mask] *= detail_gain
    output = _normalize_for_display(display_values, log_scale)
    return OperationResult(output, {"Decomposition levels": str(levels)})


def radon_transform(
    image: np.ndarray,
    angle_step: int,
    resolution: int,
    log_scale: bool,
) -> OperationResult:
    """Compute a displayable parallel-beam Radon sinogram."""

    gray = _gray(image).astype(np.float32)
    height, width = gray.shape
    scale = resolution / max(height, width)
    resized_width = max(1, min(resolution, round(width * scale)))
    resized_height = max(1, min(resolution, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        gray,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    square = np.zeros((resolution, resolution), dtype=np.float32)
    x0 = (resolution - resized_width) // 2
    y0 = (resolution - resized_height) // 2
    square[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized

    angles = np.arange(0, 180, angle_step, dtype=np.float32)
    center = ((resolution - 1) / 2.0, (resolution - 1) / 2.0)
    projections = []
    for angle in angles:
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(
            square,
            matrix,
            (resolution, resolution),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        projections.append(rotated.sum(axis=0))

    sinogram = np.stack(projections, axis=1)
    output = _normalize_for_display(sinogram, log_scale)
    return OperationResult(output, {"Projection angles": str(len(angles))})


def sobel_edges(image: np.ndarray, kernel_size: int, direction: str) -> np.ndarray:
    gray = _gray(image)
    x_gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=kernel_size)
    y_gradient = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=kernel_size)

    if direction == "Horizontal changes (dx)":
        edges = np.abs(x_gradient)
    elif direction == "Vertical changes (dy)":
        edges = np.abs(y_gradient)
    else:
        edges = cv2.magnitude(x_gradient, y_gradient)

    return cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def canny_edges(
    image: np.ndarray,
    thresholds: tuple[int, int],
    blur_kernel: int,
) -> np.ndarray:
    gray = _gray(image)
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    low, high = sorted(thresholds)
    return cv2.Canny(gray, low, high)


def binary_threshold(
    image: np.ndarray, threshold: int, inverse: bool
) -> OperationResult:
    threshold_type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, output = cv2.threshold(_gray(image), threshold, 255, threshold_type)
    foreground = 100.0 * np.count_nonzero(output) / output.size
    return OperationResult(output, {"White pixels": f"{foreground:.1f}%"})


def otsu_threshold(image: np.ndarray, inverse: bool) -> OperationResult:
    threshold_type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    selected, output = cv2.threshold(
        _gray(image), 0, 255, threshold_type | cv2.THRESH_OTSU
    )
    return OperationResult(output, {"Chosen threshold": f"{selected:.1f}"})


def equalize_histogram(image: np.ndarray, output_mode: str) -> np.ndarray:
    if output_mode == "Grayscale":
        return cv2.equalizeHist(_gray(image))

    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float,
    grid_size: int,
    output_mode: str,
) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size),
    )
    if output_mode == "Grayscale":
        return clahe.apply(_gray(image))

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def draw_contours(
    image: np.ndarray,
    thresholds: tuple[int, int],
    min_area: int,
    line_width: int,
) -> OperationResult:
    gray = cv2.GaussianBlur(_gray(image), (5, 5), 0)
    low, high = sorted(thresholds)
    edges = cv2.Canny(gray, low, high)
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    kept = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    output = image.copy()
    cv2.drawContours(output, kept, -1, (28, 238, 177), line_width, cv2.LINE_AA)

    for contour in kept:
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + width, y + height), (255, 196, 67), 1)

    largest_area = max((cv2.contourArea(contour) for contour in kept), default=0.0)
    return OperationResult(
        output,
        {
            "Contours kept": str(len(kept)),
            "Largest area": f"{largest_area:,.0f} px²",
        },
    )


def resize_image(image: np.ndarray, scale: float, interpolation: str) -> np.ndarray:
    methods = {
        "Nearest": cv2.INTER_NEAREST,
        "Linear": cv2.INTER_LINEAR,
        "Cubic": cv2.INTER_CUBIC,
        "Area": cv2.INTER_AREA,
        "Lanczos": cv2.INTER_LANCZOS4,
    }
    height, width = image.shape[:2]
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=methods[interpolation])


def rotate_image(
    image: np.ndarray,
    angle: float,
    scale: float,
    expand_canvas: bool,
) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)

    if expand_canvas:
        cosine = abs(matrix[0, 0])
        sine = abs(matrix[0, 1])
        new_width = max(1, int(height * sine + width * cosine))
        new_height = max(1, int(height * cosine + width * sine))
        matrix[0, 2] += new_width / 2.0 - center[0]
        matrix[1, 2] += new_height / 2.0 - center[1]
    else:
        new_width, new_height = width, height

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def flip_image(image: np.ndarray, direction: str) -> np.ndarray:
    flip_codes = {
        "Horizontal": 1,
        "Vertical": 0,
        "Both axes": -1,
    }
    return cv2.flip(image, flip_codes[direction])


def shear_image(image: np.ndarray, horizontal: float, vertical: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.array([[1.0, horizontal, 0.0], [vertical, 1.0, 0.0]], dtype=np.float32)
    corners = np.array(
        [[[0, 0], [width, 0], [0, height], [width, height]]],
        dtype=np.float32,
    )
    transformed = cv2.transform(corners, matrix)[0]
    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)
    matrix[:, 2] -= minimum
    output_size = (
        max(1, int(np.ceil(maximum[0] - minimum[0]))),
        max(1, int(np.ceil(maximum[1] - minimum[1]))),
    )
    return cv2.warpAffine(
        image,
        matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def add_gaussian_noise(image: np.ndarray, sigma: float, noise_mode: str) -> np.ndarray:
    generator = np.random.default_rng(42)
    noise_shape = image.shape if noise_mode == "Color" else (*image.shape[:2], 1)
    noise = generator.normal(0.0, sigma, size=noise_shape)
    return _uint8(image.astype(np.float32) + noise)


def apply_cutout(
    image: np.ndarray,
    size_ratio: float,
    location: str,
    fill: str,
) -> np.ndarray:
    height, width = image.shape[:2]
    cutout_width = max(1, round(width * size_ratio))
    cutout_height = max(1, round(height * size_ratio))

    if location == "Random (fixed seed)":
        generator = np.random.default_rng(42)
        x0 = int(generator.integers(0, max(1, width - cutout_width + 1)))
        y0 = int(generator.integers(0, max(1, height - cutout_height + 1)))
    else:
        x0 = (width - cutout_width) // 2
        y0 = (height - cutout_height) // 2

    if fill == "Image mean":
        fill_value: int | np.ndarray = np.mean(image, axis=(0, 1)).astype(np.uint8)
    elif fill == "White":
        fill_value = 255
    else:
        fill_value = 0

    output = image.copy()
    output[y0 : y0 + cutout_height, x0 : x0 + cutout_width] = fill_value
    return output


def sharpen_image(image: np.ndarray, amount: float, radius: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=radius)
    return _uint8(cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0))


def apply_operation(
    key: str, image: np.ndarray, parameters: dict[str, Any]
) -> OperationResult:
    """Apply a registered operation with its current UI parameters."""
    if key == "hsv_adjust":
        output = adjust_hsv(image, **parameters)
    elif key == "lab_adjust":
        output = adjust_lab(image, **parameters)
    elif key == "box_blur":
        output = box_blur(image, **parameters)
    elif key == "gaussian_blur":
        output = gaussian_blur(image, **parameters)
    elif key == "bilateral_filter":
        output = bilateral_filter(image, **parameters)
    elif key == "fourier_transform":
        return fourier_transform(image, **parameters)
    elif key == "dct_transform":
        return dct_transform(image, **parameters)
    elif key == "haar_wavelet_transform":
        return haar_wavelet_transform(image, **parameters)
    elif key == "radon_transform":
        return radon_transform(image, **parameters)
    elif key == "sobel_edges":
        output = sobel_edges(image, **parameters)
    elif key == "canny_edges":
        output = canny_edges(image, **parameters)
    elif key == "binary_threshold":
        return binary_threshold(image, **parameters)
    elif key == "otsu_threshold":
        return otsu_threshold(image, **parameters)
    elif key == "histogram_equalization":
        output = equalize_histogram(image, **parameters)
    elif key == "clahe":
        output = apply_clahe(image, **parameters)
    elif key == "contours":
        return draw_contours(image, **parameters)
    elif key == "resize":
        output = resize_image(image, **parameters)
    elif key == "rotate":
        output = rotate_image(image, **parameters)
    elif key == "flip":
        output = flip_image(image, **parameters)
    elif key == "shear":
        output = shear_image(image, **parameters)
    elif key == "gaussian_noise":
        output = add_gaussian_noise(image, **parameters)
    elif key == "cutout":
        output = apply_cutout(image, **parameters)
    elif key == "sharpen":
        output = sharpen_image(image, **parameters)
    else:
        raise KeyError(f"Unknown operation: {key}")

    return OperationResult(output)
