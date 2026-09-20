import unittest
import random
import sys
from unittest.mock import MagicMock, patch

class TestRNGController(unittest.TestCase):

    def test_capture_and_restore_python_random(self):
        from rng import RNGController

        rng = RNGController()
        random.seed(12345)
        val_before = random.random()

        # Capture state
        snapshot = rng.capture_state()

        # Mutate state by generating numbers
        mutated_vals = [random.random() for _ in range(5)]

        # Restore state
        snapshot.restore()
        val_after_restore = random.random()

        # The next random number should match the state immediately after capture
        random.seed(12345)
        _ = random.random()
        expected_val = random.random()
        self.assertEqual(val_after_restore, expected_val)

    def test_isolate_context_manager(self):
        from rng import RNGController

        rng = RNGController()
        random.seed(999)
        val_outside_1 = random.random()

        with rng.isolate(seed=42):
            val_inside_1 = random.random()
            val_inside_2 = random.random()

        val_outside_2 = random.random()

        # Run again from same outside seed to verify outside sequence was uninterrupted
        random.seed(999)
        _ = random.random()
        expected_outside_2 = random.random()
        self.assertEqual(val_outside_2, expected_outside_2)

        # Run inside block with seed 42 to verify determinism
        with rng.isolate(seed=42):
            self.assertEqual(random.random(), val_inside_1)
            self.assertEqual(random.random(), val_inside_2)

    def test_isolate_exception_safety(self):
        from rng import RNGController

        rng = RNGController()
        random.seed(777)
        _ = random.random()
        snapshot_before = rng.capture_state()

        try:
            with rng.isolate(seed=123):
                _ = random.random()
                raise RuntimeError("Simulated failure inside isolated block")
        except RuntimeError:
            pass

        # State after exception should match snapshot_before
        val_after = random.random()
        snapshot_before.restore()
        val_expected = random.random()
        self.assertEqual(val_after, val_expected)

    def test_defensive_cuda_handling(self):
        from rng import RNGController

        rng = RNGController()
        # Ensure capture and restore don't crash when cuda is not available
        snapshot = rng.capture_state()
        self.assertIsNone(snapshot.cuda_state)
        # Should not throw
        snapshot.restore()

    def test_rng_stream_pause_and_resume(self):
        from rng import RNGStream

        stream = RNGStream(seed=100)

        # Draw 2 numbers from stream
        with stream:
            s_val1 = random.random()
            s_val2 = random.random()

        # Draw unrelated numbers outside
        _ = [random.random() for _ in range(10)]

        # Re-enter stream, draw 3rd number
        with stream:
            s_val3 = random.random()

        # Verification: a continuous sequence from seed 100
        random.seed(100)
        e_val1 = random.random()
        e_val2 = random.random()
        e_val3 = random.random()

        self.assertEqual(s_val1, e_val1)
        self.assertEqual(s_val2, e_val2)
        self.assertEqual(s_val3, e_val3)

    def test_deterministic_model_retrofit(self):
        from transfer_learning.models import DeterministicModel

        model = DeterministicModel()
        states = model.set_rng_state(555)

        # Verify backward compatibility tuple unpacking
        old_torch, old_cuda, old_np, old_py = states
        r1 = random.random()

        # Restore
        model.restore_rng_state(states)
        r2 = random.random()

        # Test again
        random.seed(555)
        expected_r1 = random.random()
        self.assertEqual(r1, expected_r1)

    def test_build_factor_seeds(self):
        from rng import RNGController

        # Golden model: all seeds identical
        seeds_golden = RNGController.build_factor_seeds(base_seed=123)
        self.assertEqual(seeds_golden['label_choice'], 123)
        self.assertEqual(seeds_golden['sample_choice'], 123)
        self.assertEqual(seeds_golden['sample_order'], 123)
        self.assertEqual(seeds_golden['model_initialisation'], 123)
        self.assertEqual(seeds_golden['model_randomness'], 123)

        # Isolated factor: target factor gets base_seed, others get fixed_seed
        seeds_iso = RNGController.build_factor_seeds(base_seed=999, isolated_factor='sample_order', fixed_seed=42)
        self.assertEqual(seeds_iso['sample_order'], 999)
        self.assertEqual(seeds_iso['label_choice'], 42)
        self.assertEqual(seeds_iso['model_initialisation'], 42)

    def test_generate_run_seeds(self):
        import os
        import tempfile
        from rng import RNGController

        seeds_1 = RNGController.generate_run_seeds(rskf_seed=27, total_runs=10)
        self.assertEqual(len(seeds_1), 10)

        # Repeat should produce identical seeds
        seeds_2 = RNGController.generate_run_seeds(rskf_seed=27, total_runs=10)
        self.assertEqual(seeds_1, seeds_2)

        # Caching test
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "seeds.pkl")
            saved = RNGController.generate_run_seeds(rskf_seed=42, total_runs=5, cache_path=cache_file)
            self.assertTrue(os.path.exists(cache_file))

            loaded = RNGController.generate_run_seeds(rskf_seed=42, total_runs=5, cache_path=cache_file)
            self.assertEqual(saved, loaded)

    def test_seeded_random_sampler_stream_integration(self):
        from data import SeededRandomSampler

        # Mock dataset with len
        dummy_ds = [0, 1, 2, 3, 4]
        sampler1 = SeededRandomSampler(dummy_ds, seed=42)
        sampler2 = SeededRandomSampler(dummy_ds, seed=42)

        self.assertTrue(hasattr(sampler1, 'stream'))
        self.assertTrue(hasattr(sampler1, 'state'))

        # Iterating should produce identical sequence
        order1 = list(iter(sampler1))
        order2 = list(iter(sampler2))
        self.assertEqual(order1, order2)


if __name__ == "__main__":
    unittest.main()
