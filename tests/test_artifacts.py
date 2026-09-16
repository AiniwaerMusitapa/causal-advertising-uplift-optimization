"""Dependency-free snapshot checks; no training or causal-validity claim."""
import hashlib
import json
import re
import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


class ArtifactChecks(unittest.TestCase):
    def test_sample_and_feature_contract(self):
        m = load("uplift_metrics.json")
        a = load("experiment_validation.json")
        self.assertEqual(m["status"], "complete")
        self.assertEqual(m["full_rows"], a["rows"])
        self.assertEqual(m["full_rows"], 13_979_592)
        self.assertEqual(m["feature_list"], [f"f{i}" for i in range(12)])
        self.assertIs(m["post_treatment_exposure_used"], False)
        self.assertTrue(all(n > 0 for n in m["model_rows"].values()))
        self.assertLess(sum(m["model_rows"].values()), m["full_rows"])

    def test_ate_matches_observed_means(self):
        for outcome in load("ate_analysis.json")["outcomes"].values():
            self.assertAlmostEqual(outcome["ate"], outcome["treatment_mean"] - outcome["control_mean"], places=12)
            low, high = outcome["ci95_bootstrap_percentile"]
            self.assertLess(low, outcome["ate"])
            self.assertLess(outcome["ate"], high)

    def test_validation_selection(self):
        m = load("uplift_metrics.json")
        winner = max(m["validation"], key=lambda n: m["validation"][n]["qini_coefficient"])
        self.assertEqual(m["selected_model"], winner)
        self.assertEqual(m["selection_metric"], "validation Qini coefficient")
        self.assertTrue({"s_learner", "t_learner", "x_learner", "causal_forest", "response_model"}.issubset(m["test"]))

    def test_source_checksums(self):
        for name, expected in load("artifact_audit.json")["source_code_sha256"].items():
            with self.subTest(script=name):
                self.assertEqual(hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest(), expected)

    def test_eight_png_headers(self):
        figures = list((ROOT / "figures").glob("*.png"))
        self.assertEqual(len(figures), 8)
        for path in figures:
            with self.subTest(figure=path.name):
                header = path.read_bytes()[:24]
                self.assertEqual(header[:8], b"\x89PNG\r\n\x1a\n")
                width, height = struct.unpack(">II", header[16:24])
                self.assertGreaterEqual(width, 600)
                self.assertGreaterEqual(height, 400)

    def test_readme_local_links(self):
        for target in re.findall(r"\]\(([^)]+)\)", (ROOT / "README.md").read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            with self.subTest(link=target):
                self.assertTrue((ROOT / target.split("#")[0]).exists())


if __name__ == "__main__":
    unittest.main()
