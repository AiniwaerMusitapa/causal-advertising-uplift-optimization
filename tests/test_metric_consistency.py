"""Independent arithmetic checks of saved results, not raw-data replication."""
import csv
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def integrate(x, y):
    return sum((b - a) * (u + v) / 2 for a, b, u, v in zip(x, x[1:], y, y[1:]))


class MetricConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = json.loads((ROOT / 'reports/uplift_metrics.json').read_text(encoding='utf-8'))

    def test_all_curve_integrals_and_population_counts(self):
        for partition in ('validation', 'test'):
            n = self.metrics['model_rows'][partition]
            endpoints = []
            for name, result in self.metrics[partition].items():
                with self.subTest(partition=partition, model=name):
                    rows = result['curve']
                    self.assertEqual(len(rows), 100)
                    self.assertEqual(rows[-1]['fraction'], 1)
                    endpoints.append(rows[-1]['uplift'])
                    previous = 0
                    for row in rows:
                        q = row['fraction']
                        self.assertTrue(previous < q <= 1)
                        previous = q
                        self.assertEqual(row['rows'], math.ceil(q * n))
                        self.assertEqual(row['treated'] + row['control'], row['rows'])
                        self.assertGreaterEqual(row['treated'], 0)
                        self.assertGreaterEqual(row['control'], 0)
                        self.assertTrue(math.isfinite(row['uplift']))
                        self.assertAlmostEqual(row['incremental_per_population'], q * row['uplift'], places=12)
                    x = [0] + [r['fraction'] for r in rows]
                    gain = [0] + [r['incremental_per_population'] for r in rows]
                    self.assertAlmostEqual(result['auuc'], integrate(x, gain), places=12)
                    centered = [g - q * rows[-1]['uplift'] for q, g in zip(x, gain)]
                    self.assertAlmostEqual(result['qini_coefficient'], integrate(x, centered), places=12)
                    for k, at in result['uplift_at'].items():
                        match = next(r for r in rows if abs(r['fraction'] - int(k) / 100) < 1e-9)
                        self.assertEqual(at, match)
            # Targeting everyone must give the same effect regardless of ranker.
            for endpoint in endpoints:
                self.assertAlmostEqual(endpoint, endpoints[0], places=12)

    def test_deciles_partition_test_and_reconcile_full_effect(self):
        n = self.metrics['model_rows']['test']
        for name, deciles in self.metrics['test_deciles'].items():
            with self.subTest(model=name):
                self.assertEqual([d['decile'] for d in deciles], list(range(1, 11)))
                self.assertEqual(sum(d['rows'] for d in deciles), n)
                for d in deciles:
                    self.assertEqual(d['treated'] + d['control'], d['rows'])
                    self.assertLessEqual(d['ci_low'], d['ci_high'])
                    self.assertTrue(all(math.isfinite(d[k]) for k in ('uplift', 'ci_low', 'ci_high')))
                weighted = sum(d['rows'] * d['uplift'] for d in deciles) / n
                self.assertAlmostEqual(weighted, self.metrics['test'][name]['curve'][-1]['uplift'], places=12)
                self.assertAlmostEqual(deciles[0]['uplift'], self.metrics['test'][name]['uplift_at']['10']['uplift'], places=12)
        self.assertEqual(sum(self.metrics['selected_policy_segments'].values()), n)

    def test_powerbi_model_summary_matches_json(self):
        with (ROOT / 'reports/powerbi/model_summary.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), len(self.metrics['test']))
        self.assertEqual({r['model'] for r in rows}, set(self.metrics['test']))
        for row in rows:
            result = self.metrics['test'][row['model']]
            self.assertEqual(row['selected_on_validation'], str(row['model'] == self.metrics['selected_model']))
            self.assertAlmostEqual(float(row['qini']), result['qini_coefficient'], places=12)
            self.assertAlmostEqual(float(row['auuc']), result['auuc'], places=12)
            for k in ('10', '20', '30'):
                self.assertAlmostEqual(float(row[f'uplift_at_{k}']), result['uplift_at'][k]['uplift'], places=12)

    def test_summary_runs_from_another_working_directory(self):
        process = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/05_snapshot_summary.py'), '--json'],
            cwd=ROOT.parent, check=True, capture_output=True, text=True,
        )
        result = json.loads(process.stdout)
        self.assertEqual(result['selected_model'], self.metrics['selected_model'])
        self.assertEqual(result['full_records'], self.metrics['full_rows'])
        selected = self.metrics['test'][self.metrics['selected_model']]['uplift_at']['10']['uplift']
        baseline = self.metrics['test']['response_model']['uplift_at']['10']['uplift']
        self.assertAlmostEqual(result['top10_difference_percentage_points'], (selected - baseline) * 100, places=12)
        self.assertIs(result['policy_difference_significance_established'], False)
        self.assertIs(result['production_roi_established'], False)
