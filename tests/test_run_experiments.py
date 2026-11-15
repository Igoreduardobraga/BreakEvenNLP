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


class TestShots(unittest.TestCase):

    def test_it_is_evaluated_with_the_icl_shots(self):
        def shots(exp):
            cmd = exp["cmd_args"]
            return cmd[cmd.index("--num_shots") + 1]

        icl = {shots(e) for e in rx.generate_icl_commands() + rx.generate_icl_modern_commands()}
        self.assertEqual(icl, {"2"})
        for generate in (rx.generate_it_flan_t5_commands, rx.generate_it_mistral_zephyr_commands,
                         rx.generate_it_modern_commands):
            self.assertEqual({shots(e) for e in generate()}, icl, generate.__name__)


class TestDecodingFlags(unittest.TestCase):

    def test_it_gets_the_decoding_but_not_the_engine(self):
        # The tuned model is evaluated on HF, which also restricts the decoding.
        import contextlib
        import io
        argv = ["run_experiments.py", "--script", "it-modern", "--datasets", "sst2", "--models", "qwen",
                "--model-sizes", "9b", "--decoding", "guided", "--dry-run", "--logs-dir", tempfile.mkdtemp()]
        out = io.StringIO()
        with mock.patch("sys.argv", argv), contextlib.redirect_stdout(out):
            rx.main()
        commands = [line for line in out.getvalue().splitlines() if "Comando:" in line]
        self.assertTrue(commands)
        self.assertTrue(all("--decoding guided" in c and "--engine" not in c for c in commands))


class TestForce(unittest.TestCase):
    """--force must rerun finished configurations instead of letting main.py skip their folds."""

    RESULTS = os.path.join("results", "prompting", "prompting_phi_mini_vllm_guided",
                           "num_samples_1000", "sst2", "golden_model")

    def setUp(self):
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(tempfile.mkdtemp())
        for k in range(rx.DATASET_CONFIGS["sst2"]["rskf_splits"]):
            fold = os.path.join(self.RESULTS, f"repeat_0_fold_{k}")
            os.makedirs(fold)
            with open(os.path.join(fold, "results.json"), "w") as f:
                f.write("{}")

    def _run(self, *flags):
        """Runs the orchestrator with main.py mocked; returns the jobs it ran and its output."""
        import contextlib
        import io
        argv = ["run_experiments.py", "--script", "prompting-modern", "--models", "phi",
                "--datasets", "sst2", "--engine", "vllm", "--decoding", "guided",
                "--logs-dir", "logs", *flags]
        out = io.StringIO()
        with mock.patch("sys.argv", argv), contextlib.redirect_stdout(out), \
                mock.patch.object(rx, "run_job", return_value=True) as run_job:
            rx.main()
        return [c.args[0] for c in run_job.call_args_list], out.getvalue()

    def test_finished_configuration_is_skipped_without_force(self):
        jobs, _ = self._run()
        self.assertEqual(jobs, [])

    def test_force_sets_old_results_aside_and_reruns(self):
        jobs, _ = self._run("--force")
        self.assertEqual(len(jobs), 1)
        # main.py finds no folds to skip; the old ones are kept under results_old.
        self.assertFalse(os.path.exists(self.RESULTS))
        [stamp] = os.listdir("results_old")
        self.assertTrue(stamp.startswith("force_"))
        moved = os.path.join("results_old", stamp, os.path.relpath(self.RESULTS, "results"))
        self.assertTrue(os.path.exists(os.path.join(moved, "repeat_0_fold_4", "results.json")))

    def test_interrupted_force_resumes_without_force(self):
        self._run("--force")
        jobs, _ = self._run()
        self.assertEqual(len(jobs), 1)

    def test_dry_run_force_only_lists_the_move(self):
        _, out = self._run("--force", "--dry-run")
        self.assertIn(f"[SERÁ MOVIDO] {self.RESULTS} -> ", out)
        self.assertTrue(os.path.exists(os.path.join(self.RESULTS, "repeat_0_fold_0", "results.json")))
        self.assertFalse(os.path.exists("results_old"))


if __name__ == "__main__":
    unittest.main()
