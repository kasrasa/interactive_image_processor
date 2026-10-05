"""Small regression suite covering every registered operation."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.image_utils import convert_to_grayscale_rgb, create_sample_image
from src.operation_registry import OPERATION_LIST, code_snippet, default_parameters
from src.operations import (
    add_gaussian_noise,
    adjust_hsv,
    apply_operation,
    dct_transform,
    fourier_transform,
    haar_wavelet_transform,
    radon_transform,
)


class OperationSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = create_sample_image(width=320, height=220)

    def test_every_registered_operation_returns_displayable_uint8_image(self) -> None:
        for operation in OPERATION_LIST:
            with self.subTest(operation=operation.key):
                result = apply_operation(
                    operation.key,
                    self.image,
                    default_parameters(operation),
                )
                self.assertEqual(result.image.dtype, np.uint8)
                self.assertIn(result.image.ndim, (2, 3))
                self.assertGreater(result.image.shape[0], 0)
                self.assertGreater(result.image.shape[1], 0)
                if result.image.ndim == 3:
                    self.assertEqual(result.image.shape[2], 3)

    def test_hsv_scaling_clips_instead_of_wrapping(self) -> None:
        bright = np.full((8, 8, 3), 250, dtype=np.uint8)
        result = adjust_hsv(bright, hue_shift=0, saturation_scale=1.0, value_scale=1.75)
        self.assertEqual(int(result.min()), 255)
        self.assertEqual(int(result.max()), 255)

    def test_noise_preview_is_reproducible(self) -> None:
        first = add_gaussian_noise(self.image, sigma=20, noise_mode="Color")
        second = add_gaussian_noise(self.image, sigma=20, noise_mode="Color")
        np.testing.assert_array_equal(first, second)

    def test_fourier_centers_the_dc_component(self) -> None:
        constant = np.full((32, 48, 3), 120, dtype=np.uint8)

        result = fourier_transform(
            constant,
            view="Magnitude spectrum",
            center_frequency=True,
            log_scale=True,
        ).image

        self.assertEqual(np.unravel_index(np.argmax(result), result.shape), (16, 24))

    def test_full_dct_reconstruction_recovers_grayscale_input(self) -> None:
        expected = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        result = dct_transform(
            self.image,
            view="Low-frequency reconstruction",
            low_frequency_span=100,
            log_scale=True,
        ).image

        np.testing.assert_allclose(result, expected, atol=1)

    def test_wavelet_output_is_padded_for_the_selected_levels(self) -> None:
        odd_image = self.image[:219, :319]

        result = haar_wavelet_transform(
            odd_image,
            levels=3,
            detail_gain=3.0,
            log_scale=True,
        ).image

        self.assertEqual(result.shape, (224, 320))

    def test_radon_output_has_one_column_per_projection_angle(self) -> None:
        result = radon_transform(
            self.image,
            angle_step=11,
            resolution=128,
            log_scale=True,
        )

        self.assertEqual(result.image.shape, (128, 17))
        self.assertEqual(result.metrics["Projection angles"], "17")

    def test_copyable_code_matches_each_default_preview(self) -> None:
        for operation in OPERATION_LIST:
            with self.subTest(operation=operation.key):
                parameters = default_parameters(operation)
                expected = apply_operation(operation.key, self.image, parameters).image
                namespace = {"image_rgb": self.image.copy()}
                exec(code_snippet(operation.key, parameters), namespace)  # noqa: S102
                np.testing.assert_array_equal(namespace["result"], expected)

    def test_every_operation_and_snippet_supports_grayscale_input_mode(self) -> None:
        grayscale_image = convert_to_grayscale_rgb(self.image)

        for operation in OPERATION_LIST:
            with self.subTest(operation=operation.key):
                parameters = default_parameters(operation)
                expected = apply_operation(
                    operation.key,
                    grayscale_image,
                    parameters,
                ).image
                namespace = {"image_rgb": self.image.copy()}
                exec(  # noqa: S102
                    code_snippet(
                        operation.key,
                        parameters,
                        grayscale_input=True,
                    ),
                    namespace,
                )
                np.testing.assert_array_equal(namespace["result"], expected)

    def test_copyable_code_matches_alternate_branches(self) -> None:
        cases = (
            ("sobel_edges", {"kernel_size": 5, "direction": "Horizontal changes (dx)"}),
            (
                "fourier_transform",
                {
                    "view": "Phase spectrum",
                    "center_frequency": False,
                    "log_scale": False,
                },
            ),
            (
                "dct_transform",
                {
                    "view": "Low-frequency reconstruction",
                    "low_frequency_span": 35,
                    "log_scale": False,
                },
            ),
            (
                "haar_wavelet_transform",
                {"levels": 3, "detail_gain": 1.5, "log_scale": False},
            ),
            (
                "radon_transform",
                {"angle_step": 11, "resolution": 128, "log_scale": False},
            ),
            (
                "clahe",
                {"clip_limit": 3.5, "grid_size": 6, "output_mode": "Grayscale"},
            ),
            ("resize", {"scale": 1.2, "interpolation": "Lanczos"}),
            ("rotate", {"angle": -32.0, "scale": 0.9, "expand_canvas": False}),
            (
                "cutout",
                {
                    "size_ratio": 0.35,
                    "location": "Random (fixed seed)",
                    "fill": "White",
                },
            ),
            ("gaussian_noise", {"sigma": 14.0, "noise_mode": "Monochrome"}),
        )

        for key, parameters in cases:
            with self.subTest(operation=key):
                expected = apply_operation(key, self.image, parameters).image
                namespace = {"image_rgb": self.image.copy()}
                exec(code_snippet(key, parameters), namespace)  # noqa: S102
                np.testing.assert_array_equal(namespace["result"], expected)


if __name__ == "__main__":
    unittest.main()
