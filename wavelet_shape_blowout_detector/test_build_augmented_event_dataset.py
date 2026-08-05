from __future__ import annotations

import unittest

import numpy as np

from .build_augmented_event_dataset import (
    augment_values,
    draw_correlated_noise,
)


class CorrelatedResidualAugmentationTests(unittest.TestCase):
    def test_noise_is_one_contiguous_multiwheel_slice(self):
        base = np.arange(80, dtype=float)
        residuals = np.column_stack((base, 2 * base, -base, 0.5 * base))
        noise = draw_correlated_noise(
            residuals, length=20, rng=np.random.default_rng(7)
        )

        # Every adjacent row must come from the original synchronized series;
        # no wheel can receive an independently generated trajectory.
        self.assertTrue(np.allclose(np.diff(noise[:, 0]), 1.0))
        self.assertTrue(np.allclose(noise[:, 1], 2.0 * noise[:, 0]))
        self.assertTrue(np.allclose(noise[:, 2], -noise[:, 0]))
        self.assertTrue(np.allclose(noise[:, 3], 0.5 * noise[:, 0]))

    def test_augmentation_does_not_mutate_source_values(self):
        values = np.full((30, 4), 50.0)
        original = values.copy()
        residuals = np.column_stack(
            [np.sin(np.arange(100) / 10.0 + offset) for offset in range(4)]
        )
        augmented, dropout = augment_values(
            values,
            residuals,
            speed_scale=1.02,
            noise_gain=0.3,
            dropout_probability=0.0,
            max_dropout_samples=3,
            rng=np.random.default_rng(3),
        )

        self.assertEqual(dropout, 0)
        self.assertTrue(np.array_equal(values, original))
        self.assertFalse(np.array_equal(augmented, original))


if __name__ == "__main__":
    unittest.main()
