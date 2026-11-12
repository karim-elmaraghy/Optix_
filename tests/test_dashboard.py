import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import app


class DashboardTests(unittest.TestCase):
    def test_issue_2_benchmark_result_list_schema_is_loaded(self):
        item = {
            "profile_name": "default_fp16_sdpa_b1",
            "status": "success",
            "error": None,
            "requested_config": {
                "dtype": "float16",
                "attn": "sdpa",
                "batch": 1,
                "compile": False,
                "max_new_tokens": 128,
            },
            "actual_config": {
                "dtype": "float16",
                "attn": "sdpa",
                "batch": 1,
                "compile": False,
                "max_new_tokens": 128,
            },
            "repeats": 3,
            "total_time_sec": {"median": 10.0, "min": 9.0, "max": 11.0},
            "latency_p50_sec": 1.0,
            "latency_p95_sec": 1.2,
            "docs_per_sec": 2.0,
            "tokens_per_sec": 20.0,
            "joules_per_doc": 3.0,
            "idle_power_w": 5.0,
            "compile_time_sec": None,
            "noisy": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "benchmark_20261104T000000.json"
            path.write_text(json.dumps([item]), encoding="utf-8")
            loaded = app._load_benchmark(path)

        self.assertEqual(loaded, [item])

    def test_empty_results_are_explicitly_not_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            sample = results / "sample"
            sample.mkdir()
            (sample / "benchmark_sample.json").write_text("[]", encoding="utf-8")
            (sample / "agent_log_sample.json").write_text('{"steps":[]}', encoding="utf-8")

            data = app.load_dashboard_data(results)
            view = app.build_dashboard_view(data)

        self.assertIn("SAMPLE DATA - NOT REAL RESULTS", view["banner"])
        self.assertIn("No real benchmark result set", view["banner"])
        self.assertTrue(all(row[6] == "Not run" for row in view["profile_rows"]))
        self.assertFalse(data["has_real_benchmarks"])
        self.assertIn("fp16_sdpa_b4_compile", [row[0] for row in view["profile_rows"]])

    def test_real_json_values_and_reasons_are_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            benchmark = [
                {
                    "profile_name": "naive_fp32_eager_b1",
                    "status": "success",
                    "requested_config": {"dtype": "float32"},
                    "docs_per_sec": 1.25,
                    "joules_per_doc": 4.5,
                    "latency_p50_sec": 0.8,
                    "latency_p95_sec": 1.1,
                },
                {
                    "profile_name": "default_fp16_sdpa_b1",
                    "status": "success",
                    "requested_config": {"dtype": "float16"},
                    "docs_per_sec": 2.5,
                    "joules_per_doc": 2.0,
                    "latency_p50_sec": 0.4,
                    "latency_p95_sec": 0.6,
                },
                {
                    "profile_name": "candidate",
                    "status": "failed",
                    "error": "measured backend failure",
                    "requested_config": {"dtype": "float16", "batch": 4},
                },
                {
                    "profile_name": "rejected_candidate",
                    "status": "success",
                    "requested_config": {"dtype": "float16", "batch": 8},
                    "docs_per_sec": 2.4,
                },
            ]
            (results / "benchmark_real.json").write_text(json.dumps(benchmark), encoding="utf-8")
            (results / "agent_log.json").write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "config": {"dtype": "float16", "batch": 4},
                                "hypothesis": "Try a larger batch",
                                "source": "rule_fallback",
                                "decision": "REJECT",
                                "reason": "Candidate failed during execution",
                            },
                            {
                                "config": {"dtype": "float16", "batch": 8},
                                "hypothesis": "Try a different batch",
                                "source": "muse_api",
                                "decision": "REJECT",
                                "reason": "measured validator rejection",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            data = app.load_dashboard_data(results)
            view = app.build_dashboard_view(data)

        self.assertEqual(view["profile_rows"][2][6], "Failed")
        self.assertEqual(view["profile_rows"][2][7], "measured backend failure")
        self.assertIn("Rule fallback", view["agent_rows"][0][1])
        self.assertIn("measured backend failure", view["rejection_rows"][0][2])
        self.assertIn("measured validator rejection", view["rejection_rows"][1][2])
        self.assertEqual(view["comparison_rows"][0][1], "Not recorded")
        self.assertFalse(data["benchmark_is_sample"])

    def test_agent_sources_are_distinguished(self):
        self.assertEqual(app._source_label({"source": "muse_api"}), "Meta API")
        self.assertEqual(app._source_label({"source": "meta_api"}), "Meta API")
        self.assertEqual(
            app._source_label({"source": "rule_fallback"}),
            "Rule fallback (not Meta API)",
        )

    def test_agent_b_jsonl_promotions_use_measured_results(self):
        config = {
            "dtype": "float16",
            "attn": "sdpa",
            "batch": 4,
            "compile": False,
            "max_new_tokens": 128,
        }
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            (results / "benchmark_real.json").write_text(
                json.dumps(
                    [
                        {
                            "profile_name": "default_fp16_sdpa_b1",
                            "status": "success",
                            "requested_config": {"dtype": "float16", "batch": 1},
                            "docs_per_sec": 2.0,
                        },
                        {
                            "profile_name": "candidate_b4",
                            "status": "success",
                            "requested_config": config,
                            "actual_config": config,
                            "docs_per_sec": 3.0,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            step = {
                "proposal": {
                    "config": config,
                    "reason": "Try batching",
                    "source": "meta_api",
                },
                "result": {
                    "benchmark": {
                        "profile_name": "candidate_b4",
                        "status": "success",
                        "requested_config": config,
                        "actual_config": config,
                        "docs_per_sec": 3.0,
                    },
                    "validation": {
                        "accepted": True,
                        "docs": [{"token_f1": 0.9, "rouge_l": 0.8, "reasons": []}],
                    },
                },
                "accepted": True,
                "reasons": [],
            }
            (results / "agent_log.json").write_text(
                json.dumps(step) + "\n",
                encoding="utf-8",
            )
            (results / "dashboard_apply_log.jsonl").write_text(
                json.dumps(
                    {
                        "candidate_config": config,
                        "apply_result": {
                            "action": "promoted",
                            "active_config": config,
                            "validation": {
                                "accepted": True,
                                "split": "heldout",
                                "docs": [
                                    {"token_f1": 0.95, "rouge_l": 0.85, "reasons": []}
                                ],
                                "reasons": [],
                            },
                        },
                        "benchmark": {"docs_per_sec": 2.8, "joules_per_doc": 1.1},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            data = app.load_dashboard_data(results)
            view = app.build_dashboard_view(data)

        self.assertIn("Meta API", view["agent_rows"][0][1])
        self.assertEqual(view["winner_markdown"].splitlines()[0], "**Chosen profile:** candidate_b4")
        candidate_row = next(row for row in view["profile_rows"] if row[0] == "candidate_b4")
        self.assertEqual(candidate_row[6], "Accepted")
        self.assertEqual(candidate_row[3], "1.50x")
        self.assertEqual(candidate_row[4], "0.900")
        self.assertEqual(candidate_row[5], "0.800")
        self.assertIn("Held-out confirmation: accepted", view["winner_markdown"])
        self.assertIn("0.950", view["winner_markdown"])
        self.assertIn("0.850", view["winner_markdown"])
        self.assertIn("2.800 docs/s", view["winner_markdown"])
        self.assertIn("1.100 J/doc", view["winner_markdown"])

    def test_safe_apply_fails_closed_without_real_results(self):
        self.assertIn("Not applied", app.safe_apply(None))

    def test_safe_apply_does_not_claim_activation_without_runtime_callback(self):
        response = app.safe_apply(
            {
                "has_real_benchmarks": True,
                "chosen": {"profile_name": "candidate"},
            }
        )
        self.assertIn("no active inference runtime", response)
        self.assertIn("no profile was activated or rolled back", response)

    def test_successful_benchmark_without_validator_decision_is_measured_only(self):
        status, reason = app._status_and_reason(
            {"status": "success", "docs_per_sec": 1.2},
            None,
        )
        self.assertEqual(status, "Measured (not validated)")
        self.assertIn("no quality decision recorded", reason)

    def test_accepted_step_without_successful_benchmark_is_unverified(self):
        status, reason = app._status_and_reason(
            {"profile_name": "candidate"},
            {"decision": "PROMOTE", "reason": "proposal accepted"},
        )
        self.assertEqual(status, "Unverified")
        self.assertIn("no successful benchmark", reason)

    def test_failed_candidate_cannot_be_selected_as_winner(self):
        config = {"dtype": "float16", "batch": 4}
        failed = {
            "profile_name": "failed_candidate",
            "status": "failed",
            "requested_config": config,
            "error": "backend error",
        }
        selected_step = {
            "config": config,
            "decision": "PROMOTE",
            "chosen": True,
            "benchmark": failed,
        }
        self.assertIsNone(app._find_chosen([failed], [selected_step]))

    def test_module_import_does_not_require_gpu_or_ui_packages(self):
        check = (
            "import sys, app; "
            "assert not {'torch', 'transformers', 'datasets', 'gradio'} & sys.modules.keys()"
        )
        completed = subprocess.run(
            [sys.executable, "-c", check],
            cwd=Path(app.__file__).parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
