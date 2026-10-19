import unittest

import numpy as np

from asie_engine.core import NonLinearInvariantCausalEngine


class TestPredictiveEngineValidation(unittest.TestCase):
    def test_invalid_constructor_values_are_rejected(self):
        invalid = [
            {"num_signals": 0},
            {"num_signals": True},
            {"num_signals": 2.5},
            {"num_signals": 2, "max_lags": 0},
            {"num_signals": 2, "train_window": 1},
            {"num_signals": 2, "latent_factors": -1},
            {"num_signals": 2, "learning_rate": float("nan")},
            {"num_signals": 2, "learning_rate": 0},
            {"num_signals": 2, "epochs": False},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    NonLinearInvariantCausalEngine(**kwargs)

    def test_bad_observations_are_rejected_without_buffer_mutation(self):
        engine = NonLinearInvariantCausalEngine(2, train_window=5, max_lags=2)
        for observation in (
            [1.0],
            [[1.0, 2.0]],
            [1.0, float("nan")],
            [1.0, float("inf")],
            ["not", "numeric"],
        ):
            with self.subTest(observation=observation):
                with self.assertRaises(ValueError):
                    engine.step(observation)
        self.assertEqual(engine.buffer, [])

    def test_output_uses_target_rows_and_source_columns(self):
        engine = NonLinearInvariantCausalEngine(
            3, train_window=5, max_lags=2, latent_factors=0, epochs=1, random_state=7
        )
        rng = np.random.default_rng(99)
        result = None
        for row in rng.normal(size=(7, 3)):
            result = engine.step(row)
        self.assertIsNotNone(result)
        weights, prediction = result
        self.assertEqual(weights.shape, (3, 3))
        self.assertEqual(prediction.shape, (3,))
        np.testing.assert_array_equal(np.diag(weights), np.zeros(3))
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(len(engine.historical_predictive_weight_matrices), 1)

    def test_seeded_engines_are_reproducible(self):
        observations = np.random.default_rng(123).normal(size=(7, 2))
        kwargs = dict(
            num_signals=2, train_window=5, max_lags=2,
            latent_factors=0, epochs=1, random_state=19
        )
        first = NonLinearInvariantCausalEngine(**kwargs)
        second = NonLinearInvariantCausalEngine(**kwargs)
        result_a = result_b = None
        for row in observations:
            result_a = first.step(row)
            result_b = second.step(row)
        np.testing.assert_allclose(result_a[0], result_b[0])
        np.testing.assert_allclose(result_a[1], result_b[1])

    def test_step_does_not_reset_global_numpy_rng(self):
        np.random.seed(2026)
        expected_next = np.random.random()
        np.random.seed(2026)
        engine = NonLinearInvariantCausalEngine(
            2, train_window=5, max_lags=2, latent_factors=0, epochs=1, random_state=3
        )
        for row in np.random.default_rng(5).normal(size=(7, 2)):
            engine.step(row)
        actual_next = np.random.random()
        self.assertEqual(actual_next, expected_next)

    def test_zero_norm_invariance_is_not_reported_as_perfect(self):
        engine = NonLinearInvariantCausalEngine(2)
        engine.historical_predictive_weight_matrices = [
            np.zeros((2, 2)),
            np.zeros((2, 2)),
        ]
        self.assertEqual(engine._compute_structural_invariance(), 0.0)


if __name__ == "__main__":
    unittest.main()
