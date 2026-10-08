import time
import unittest


class TestSustainabilityTracker(unittest.TestCase):

    def test_start_stop_report(self):
        from sustainability import SustainabilityTracker
        with SustainabilityTracker(carbon_intensity=0.074) as t:
            time.sleep(0.05)
        r = t.report
        self.assertGreaterEqual(r.duration_seconds, 0.05)
        self.assertGreaterEqual(r.energy_kwh, 0.0)
        self.assertAlmostEqual(r.co2_kg, r.energy_kwh * 0.074)
        self.assertTrue(r.hardware)
        self.assertEqual(r.carbon_intensity, 0.074)

    def test_stop_before_start_raises(self):
        from sustainability import SustainabilityTracker
        with self.assertRaises(RuntimeError):
            SustainabilityTracker().stop()

    def test_env_carbon_intensity(self):
        import os
        from sustainability import SustainabilityTracker
        os.environ["CARBON_INTENSITY_KG_PER_KWH"] = "0.1"
        try:
            t = SustainabilityTracker().start()
            r = t.stop()
            self.assertEqual(r.carbon_intensity, 0.1)
        finally:
            del os.environ["CARBON_INTENSITY_KG_PER_KWH"]


class TestSustainabilityResultStore(unittest.TestCase):

    def test_record_fold_persists_sustainability(self):
        import shutil
        import tempfile
        from result_store import ResultStore
        d = tempfile.mkdtemp()
        try:
            store = ResultStore(
                results_path=d, storage_format="csv",
                experiment_name="e", experiment_type="prompting",
                model_name="m", dataset="sst2", factor="golden_model",
                configuration_name="c",
            )
            store.record_fold(
                repeat=0, fold=0, run_seed_used=1,
                randomness_factor_seeds={"label_choice": 1},
                metrics={"f1_macro": 0.5},
                real=[0, 1], predicted=[0, 1],
                duration_seconds=2.0,
                energy_kwh=0.001, co2_kg=0.0004,
                hardware="TEST-GPU", carbon_intensity=0.4,
            )
            rows = store.load_summary(as_dataframe=False)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["energy_kwh"], 0.001)
            self.assertAlmostEqual(rows[0]["co2_kg"], 0.0004)
            self.assertEqual(rows[0]["hardware"], "TEST-GPU")
            details = store.load_fold_details(0, 0)
            self.assertEqual(details["sustainability"]["hardware"], "TEST-GPU")
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
