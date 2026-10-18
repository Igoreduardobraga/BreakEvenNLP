"""Grouping of orchestrator commands into one main.py --sweep process."""
import argparse
import json
import os
import tempfile
import unittest
from unittest import mock

import run_experiments as rx


def _exp(exp_type, model, dataset, n, splits=5, extra=()):
    cmd = ["python", "main.py", "--experiment_type", exp_type, "--model", model,
           "--dataset", dataset, "--num_labelled", str(n),
           "--configuration_name", f"num_samples_{n}", "--rskf_splits", str(splits), *extra]
    return {"script_origin": "test", "experiment_type": exp_type, "model": model, "model_size": "9b",
            "dataset": dataset, "num_labelled": n, "cmd_args": cmd}


class TestBuildJobs(unittest.TestCase):

    def test_icl_configs_of_one_model_share_a_process(self):
        exps = [_exp("icl", "qwen", "sst2", 10), _exp("icl", "qwen", "boolq", 50, splits=10)]
        [job] = rx.build_jobs(exps, group=True)
        self.assertEqual(job["sweep"], [
            {"dataset": "sst2", "num_labelled": 10, "configuration_name": "num_samples_10", "rskf_splits": 5},
            {"dataset": "boolq", "num_labelled": 50, "configuration_name": "num_samples_50", "rskf_splits": 10},
        ])
        cmd = rx.job_command(job, "logs")
        for flag in rx.SWEEP_FLAGS:
            self.assertNotIn(flag, cmd)
        self.assertEqual(cmd[-2], "--sweep")

    def test_different_models_or_settings_are_not_merged(self):
        exps = [_exp("icl", "qwen", "sst2", 10), _exp("icl", "phi", "sst2", 10),
                _exp("icl", "qwen", "sst2", 50, extra=("--num_shots", "5"))]
        jobs = rx.build_jobs(exps, group=True)
        self.assertEqual(len(jobs), 3)
        self.assertTrue(all("sweep" not in j for j in jobs))  # singletons run as plain commands

    def test_finetuning_and_no_group_stay_one_process_each(self):
        ft = [_exp("finetuning", "bert", "sst2", n) for n in (10, 50)]
        self.assertEqual(len(rx.build_jobs(ft, group=True)), 2)
        icl = [_exp("icl", "qwen", "sst2", n) for n in (10, 50)]
        self.assertEqual(len(rx.build_jobs(icl, group=False)), 2)

    def test_run_job_writes_sweep_file_passed_to_main(self):
        exps = [_exp("icl", "qwen", "sst2", 10), _exp("icl", "qwen", "sst2", 50)]
        [job] = rx.build_jobs(exps, group=True)
        logs = tempfile.mkdtemp()
        with mock.patch.object(rx.subprocess, "Popen") as popen:
            popen.return_value.wait.return_value = 0
            self.assertTrue(rx.run_job(job, logs_dir=logs))
        cmd = popen.call_args.args[0]
        with open(cmd[cmd.index("--sweep") + 1]) as f:
            self.assertEqual(json.load(f), job["sweep"])


class TestMainSweep(unittest.TestCase):

    def _parser(self):
        p = argparse.ArgumentParser()
        p.add_argument("--dataset", choices=["sst2", "boolq"])
        return p

    def _write(self, configs):
        path = os.path.join(tempfile.mkdtemp(), "sweep.json")
        with open(path, "w") as f:
            json.dump(configs, f)
        return path

    def test_load_sweep_accepts_known_keys(self):
        import main
        main.args = argparse.Namespace(dataset="sst2")
        configs = [{"dataset": "boolq", "num_labelled": 10}, {"num_labelled": 50}]
        self.assertEqual(main.load_sweep(self._write(configs), self._parser()), configs)

    def test_load_sweep_rejects_unknown_keys_and_datasets(self):
        import main
        main.args = argparse.Namespace(dataset="sst2")
        for bad in ([{"model": "phi"}], [{"dataset": "imdb"}]):
            with self.assertRaises(SystemExit):
                main.load_sweep(self._write(bad), self._parser())

    def test_failed_configuration_does_not_stop_the_sweep(self):
        import main
        ran = []

        def fake_run(config):
            ran.append(config["num_labelled"])
            if config["num_labelled"] == 10:
                raise RuntimeError("boom")

        sweep = self._write([{"num_labelled": 10}, {"num_labelled": 50}])
        argv = ["main.py", "--experiment_type", "finetuning", "--model", "bert", "--sweep", sweep]
        with mock.patch.object(main, "run_configuration", side_effect=fake_run), \
                mock.patch("sys.argv", argv), \
                mock.patch.object(main.torch, "device"):
            with self.assertRaises(SystemExit) as ctx:
                main.main()
        self.assertEqual(ctx.exception.code, 1)
        self.assertEqual(ran, [10, 50])


if __name__ == "__main__":
    unittest.main()
