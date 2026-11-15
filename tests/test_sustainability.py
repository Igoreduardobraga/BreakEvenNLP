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


class TestEnergySources(unittest.TestCase):

    def test_integrator_trapezoid(self):
        from sustainability import _Integrator
        readings = iter([100.0, 200.0, 200.0])
        integ = _Integrator(lambda: next(readings))
        for t in (0.0, 1.0, 3.0):
            integ.sample(t)
        self.assertAlmostEqual(integ.joules, 150.0 + 400.0)
        self.assertEqual(integ.samples, 3)

    def test_rapl_reads_packages_only_and_handles_wrap(self):
        import os
        import tempfile
        from sustainability import _RaplCpu
        root = tempfile.mkdtemp()

        def domain(name, energy, max_range=1000):
            d = os.path.join(root, name)
            os.makedirs(d)
            open(os.path.join(d, "energy_uj"), "w").write(str(energy))
            open(os.path.join(d, "max_energy_range_uj"), "w").write(str(max_range))
            return d

        pkg = domain("intel-rapl:0", 900)
        domain("intel-rapl:0:0", 5)  # core sub-domain: must not be double counted
        rapl = _RaplCpu(root)
        self.assertEqual(len(rapl.domains), 1)
        start = rapl.snapshot()
        open(os.path.join(pkg, "energy_uj"), "w").write("100")
        self.assertAlmostEqual(rapl.joules_since(start), 200 / 1e6)

    def test_rapl_missing_raises(self):
        import tempfile
        from sustainability import _RaplCpu
        with self.assertRaises(OSError):
            _RaplCpu(tempfile.mkdtemp())

    def test_nvidia_smi_sums_only_visible_gpus(self):
        from unittest import mock
        import sustainability as sm
        out = "0, GPU-aaa, 100.0\n1, GPU-bbb, 300.0\n"
        fake = mock.Mock(returncode=0, stdout=out)
        with mock.patch.object(sm.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch.object(sm.subprocess, "run", return_value=fake):
            self.assertEqual(sm._nvidia_smi_power_fn(["bbb"])(), 300.0)
            self.assertEqual(sm._nvidia_smi_power_fn(["0"])(), 100.0)
            self.assertIsNone(sm._nvidia_smi_power_fn(["ccc"])())


class _FakeNvml:
    def __init__(self, energy_j):
        self.energy = iter(energy_j)

    def energy_joules(self):
        return next(self.energy)

    def power_watts(self):
        return None

    def close(self):
        pass


class TestTrackerSourceSelection(unittest.TestCase):

    def _patched(self, gpus, nvml=None):
        from unittest import mock
        import sustainability as sm
        patches = [
            mock.patch.object(sm, "_cuda_device_count", return_value=len(gpus)),
            mock.patch.object(sm, "_visible_gpus", return_value=gpus),
            mock.patch.object(sm, "_RaplCpu", side_effect=OSError),
            mock.patch.object(sm, "_cpu_utilization_fn", return_value=lambda: 0.5),
        ]
        if nvml is not None:
            patches.append(mock.patch.object(sm, "_NvmlGpus", return_value=nvml))
        else:
            patches.append(mock.patch.object(sm, "_NvmlGpus", side_effect=ImportError))
            patches.append(mock.patch.object(sm, "_nvidia_smi_power_fn", return_value=None))
        return patches

    def _run(self, patches, **kw):
        from contextlib import ExitStack
        from sustainability import SustainabilityTracker
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            t = SustainabilityTracker(carbon_intensity=0.1, sample_seconds=0.01, **kw).start()
            time.sleep(0.05)
            return t.stop()

    def test_nvml_energy_counter_is_preferred(self):
        r = self._run(self._patched(["0"], nvml=_FakeNvml([1000.0, 4600.0])))
        self.assertEqual(r.method, "gpu:nvml-energy;cpu:utilization-tdp")
        self.assertAlmostEqual(r.gpu_energy_kwh, 3600.0 / 3_600_000)
        self.assertAlmostEqual(r.energy_kwh, r.gpu_energy_kwh + r.cpu_energy_kwh)
        self.assertAlmostEqual(r.co2_kg, r.energy_kwh * 0.1)

    def test_cpu_estimate_uses_tdp_times_utilization(self):
        import os
        os.environ["CPU_TDP_WATTS"] = "100"
        try:
            r = self._run(self._patched([]))
        finally:
            del os.environ["CPU_TDP_WATTS"]
        self.assertEqual(r.method, "gpu:none;cpu:utilization-tdp")
        self.assertEqual(r.gpu_energy_kwh, 0.0)
        # 100 W x 50% over the measured window
        self.assertAlmostEqual(r.cpu_energy_kwh * 3_600_000 / r.duration_seconds, 50.0, delta=5.0)

    def test_tdp_fallback_without_any_gpu_reader(self):
        from sustainability import FALLBACK_GPU_TDP_WATTS
        r = self._run(self._patched(["0"]))
        self.assertEqual(r.method, "gpu:tdp-fallback;cpu:utilization-tdp")
        self.assertAlmostEqual(r.gpu_energy_kwh * 3_600_000 / r.duration_seconds,
                               FALLBACK_GPU_TDP_WATTS, delta=1.0)


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
                gpu_energy_kwh=0.0007, cpu_energy_kwh=0.0003,
                energy_method="gpu:nvml-energy;cpu:rapl",
            )
            rows = store.load_summary(as_dataframe=False)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["energy_kwh"], 0.001)
            self.assertAlmostEqual(rows[0]["co2_kg"], 0.0004)
            self.assertEqual(rows[0]["hardware"], "TEST-GPU")
            self.assertAlmostEqual(rows[0]["gpu_energy_kwh"], 0.0007)
            self.assertEqual(rows[0]["energy_method"], "gpu:nvml-energy;cpu:rapl")
            details = store.load_fold_details(0, 0)
            self.assertEqual(details["sustainability"]["hardware"], "TEST-GPU")
            self.assertEqual(details["sustainability"]["method"], "gpu:nvml-energy;cpu:rapl")
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
