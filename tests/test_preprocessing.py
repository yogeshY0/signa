"""Tests for ink-polarity normalization in Stage 2 preprocessing."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from preprocessing.signature_preprocessing import to_grayscale


def _dark_ink_on_white() -> np.ndarray:
    """White(255) canvas with a dark(0) band."""
    image = np.full((64, 128), 255, np.uint8)
    image[20:44, 30:98] = 0
    return image


def _light_ink_on_dark() -> np.ndarray:
    """Dark(10) canvas with a light(250) band (inverted polarity)."""
    image = np.full((64, 128), 10, np.uint8)
    image[20:44, 30:98] = 250
    return image


class PolarityTests(unittest.TestCase):
    def _assert_ink_dark_output(self, out: np.ndarray) -> None:
        # Ink band is dark, background is light (>200).
        self.assertTrue(np.all(np.less_equal(out[20:44, 30:98], 64)))
        self.assertTrue(np.all(np.greater(out[:10, :], 200)))

    def _assert_ink_light_output(self, out: np.ndarray) -> None:
        # Ink band is light, background is dark: polarity was preserved.
        self.assertTrue(np.all(np.greater_equal(out[20:44, 30:98], 200)))
        self.assertTrue(np.all(np.less(out[:10, :], 100)))

    def test_auto_keeps_dark_ink_on_white(self) -> None:
        self._assert_ink_dark_output(to_grayscale(_dark_ink_on_white(), remove_shadow=False))

    def test_auto_inverts_light_ink_on_dark(self) -> None:
        out = to_grayscale(_light_ink_on_dark(), remove_shadow=False)
        # After polarity resolution the former light band becomes the (dark) ink.
        self._assert_ink_dark_output(out)
        self.assertEqual(out.max(), 255)

    def test_explicit_override_does_not_auto_detect(self) -> None:
        # Forced "dark" keeps an inverted image as-is (band stays light).
        self._assert_ink_light_output(
            to_grayscale(_light_ink_on_dark(), ink="dark", remove_shadow=False)
        )
        # Forced "light" inverts a normal dark-on-white image (band becomes light).
        self._assert_ink_light_output(
            to_grayscale(_dark_ink_on_white(), ink="light", remove_shadow=False)
        )


if __name__ == "__main__":
    unittest.main()