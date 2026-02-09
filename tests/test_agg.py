from __future__ import annotations

import unittest

import torch

from fedsvp.algorithms.agg import weighted_average_state_dict


class WeightedAverageStateDictTests(unittest.TestCase):
    def test_float_tensors_match_manual_weighted_average(self):
        states = [
            {
                "w": torch.tensor([1.0, 3.0], dtype=torch.float32),
                "b": torch.tensor([2.0], dtype=torch.float32),
            },
            {
                "w": torch.tensor([3.0, 5.0], dtype=torch.float32),
                "b": torch.tensor([6.0], dtype=torch.float32),
            },
        ]
        out = weighted_average_state_dict(states, [1.0, 3.0])

        expected_w = torch.tensor([2.5, 4.5], dtype=torch.float32)
        expected_b = torch.tensor([5.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(out["w"], expected_w, atol=1e-6))
        self.assertTrue(torch.allclose(out["b"], expected_b, atol=1e-6))

    def test_long_buffer_uses_reference_client_value(self):
        states = [
            {
                "w": torch.tensor([1.0], dtype=torch.float32),
                "num_batches_tracked": torch.tensor(10, dtype=torch.long),
            },
            {
                "w": torch.tensor([3.0], dtype=torch.float32),
                "num_batches_tracked": torch.tensor(20, dtype=torch.long),
            },
            {
                "w": torch.tensor([5.0], dtype=torch.float32),
                "num_batches_tracked": torch.tensor(30, dtype=torch.long),
            },
        ]
        out = weighted_average_state_dict(states, [0.2, 0.5, 0.3])

        self.assertTrue(torch.allclose(out["w"], torch.tensor([3.2], dtype=torch.float32), atol=1e-6))
        self.assertEqual(int(out["num_batches_tracked"].item()), 20)
        self.assertEqual(out["num_batches_tracked"].dtype, torch.long)

    def test_bool_buffer_uses_reference_client_value(self):
        states = [
            {
                "w": torch.tensor([1.0], dtype=torch.float32),
                "flag": torch.tensor([True], dtype=torch.bool),
            },
            {
                "w": torch.tensor([2.0], dtype=torch.float32),
                "flag": torch.tensor([False], dtype=torch.bool),
            },
        ]
        out = weighted_average_state_dict(states, [0.5, 0.5])

        # tie -> first argmax index (client 0)
        self.assertTrue(bool(out["flag"].item()))
        self.assertEqual(out["flag"].dtype, torch.bool)

    def test_key_mismatch_raises_clear_error(self):
        states = [
            {
                "a": torch.tensor([1.0], dtype=torch.float32),
                "b": torch.tensor([2.0], dtype=torch.float32),
            },
            {
                "a": torch.tensor([3.0], dtype=torch.float32),
                "c": torch.tensor([4.0], dtype=torch.float32),
            },
        ]
        with self.assertRaisesRegex(ValueError, r"state_dict keys mismatch at index 1"):
            weighted_average_state_dict(states, [0.4, 0.6])


if __name__ == "__main__":
    unittest.main()
