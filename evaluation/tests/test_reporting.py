import json
import tempfile
import unittest
from pathlib import Path

from evaluation.metrics import METRIC_NAMES
from evaluation.reporting import write_report_artifacts


class ReportingTest(unittest.TestCase):
    def test_writes_stable_artifact_names_without_plots(self):
        metrics = {
            name: (10.0 if name.endswith("latency_ms") else 1.0)
            for name in METRIC_NAMES
        }
        details = [
            {
                "id": "case_001",
                "category": "SQL Injection",
                "difficulty": "Easy",
                "question": "Question",
                "status": "completed",
                "metrics": metrics,
                "errors": [],
            }
        ]
        summary = {
            "run_id": "test",
            "generated_at": "now",
            "benchmark_sha256": "abc",
            "completed_cases": 1,
            "selected_cases": 1,
            "runtime_error_cases": 0,
            "overall_metrics": metrics,
            "metrics_by_category": {
                "SQL Injection": {"case_count": 1, "metrics": metrics}
            },
            "metrics_by_difficulty": {
                "Easy": {"case_count": 1, "metrics": metrics}
            },
            "error_counts": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            artifacts = write_report_artifacts(
                summary, details, output, make_plots=False
            )
            self.assertTrue(Path(artifacts["summary"]).exists())
            self.assertTrue(Path(artifacts["details"]).exists())
            self.assertTrue(Path(artifacts["report"]).exists())
            payload = json.loads((output / "benchmark_summary.json").read_text())
            self.assertEqual(payload["completed_cases"], 1)
            self.assertIn("## Overall Metrics", (output / "benchmark_report.md").read_text())


if __name__ == "__main__":
    unittest.main()

