"""Unit tests for the app's presentation-independent helpers."""

from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from src.explorer import (
    format_parameter_value,
    image_as_png,
    luminance_histogram,
    sweep_variants,
)
from src.operation_registry import OPERATION_LIST, default_parameters


class ImageProcessingHelperTests(unittest.TestCase):
    def test_image_as_png_round_trips_without_changing_pixels(self) -> None:
        image = np.array(
            [[[0, 64, 255], [128, 32, 16]], [[255, 255, 255], [0, 0, 0]]],
            dtype=np.uint8,
        )

        decoded = np.asarray(Image.open(BytesIO(image_as_png(image))))

        np.testing.assert_array_equal(decoded, image)

    def test_luminance_histogram_is_normalized(self) -> None:
        image = np.zeros((4, 5), dtype=np.uint8)

        histogram = luminance_histogram(image)

        self.assertEqual(histogram.shape, (256,))
        self.assertEqual(histogram[0], 1.0)
        self.assertAlmostEqual(float(histogram.sum()), 1.0)

    def test_parameter_values_are_formatted_for_display(self) -> None:
        self.assertEqual(format_parameter_value((10, 25)), "10 – 25")
        self.assertEqual(format_parameter_value(True), "On")
        self.assertEqual(format_parameter_value(1.5), "1.5")

    def test_sweep_preserves_other_parameters(self) -> None:
        operation = next(item for item in OPERATION_LIST if item.key == "hsv_adjust")
        parameters = default_parameters(operation)

        comparison = sweep_variants(operation, parameters)

        self.assertIsNotNone(comparison)
        control, variants = comparison
        self.assertEqual(control.key, "saturation_scale")
        self.assertEqual(len(variants), 3)
        self.assertEqual(variants[1][1]["saturation_scale"], 1.0)
        for _, variant in variants:
            self.assertEqual(variant["hue_shift"], parameters["hue_shift"])
            self.assertEqual(variant["value_scale"], parameters["value_scale"])


if __name__ == "__main__":
    unittest.main()
