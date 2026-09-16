"""Read the committed results without data downloads or ML dependencies."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize():
    metrics = json.loads((ROOT / 'reports/uplift_metrics.json').read_text(encoding='utf-8'))
    ate = json.loads((ROOT / 'reports/ate_analysis.json').read_text(encoding='utf-8'))
    selected = metrics['selected_model']
    causal = metrics['test'][selected]['uplift_at']['10']['uplift']
    response = metrics['test']['response_model']['uplift_at']['10']['uplift']
    visit = ate['outcomes']['visit']
    return {
        'scope': 'Committed offline result snapshot; not a new training run',
        'completed_at_utc': metrics['completed_at_utc'],
        'full_records': metrics['full_rows'],
        'model_test_records': metrics['model_rows']['test'],
        'selected_model': selected,
        'selection_metric': metrics['selection_metric'],
        'visit_ate_percentage_points': visit['ate'] * 100,
        'visit_ate_bootstrap_ci95_percentage_points': [
            x * 100 for x in visit['ci95_bootstrap_percentile']
        ],
        'top10_causal_uplift_percent': causal * 100,
        'top10_response_uplift_percent': response * 100,
        'top10_difference_percentage_points': (causal - response) * 100,
        'policy_difference_significance_established': False,
        'production_roi_established': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Print machine-readable results')
    args = parser.parse_args()
    result = summarize()
    if args.json:
        print(json.dumps(result, indent=2))
        return
    low, high = result['visit_ate_bootstrap_ci95_percentage_points']
    print(result['scope'])
    print(f"Recorded run: {result['completed_at_utc']}")
    print(f"Experiment records: {result['full_records']:,}")
    print(f"Held-out model test records: {result['model_test_records']:,}")
    print(f"Selected: {result['selected_model']} ({result['selection_metric']})")
    print(f"Visit ATE: {result['visit_ate_percentage_points']:+.4f} pp; "
          f"95% bootstrap CI [{low:.4f}, {high:.4f}] pp")
    print(f"Top-10% causal / response uplift: "
          f"{result['top10_causal_uplift_percent']:.4f}% / "
          f"{result['top10_response_uplift_percent']:.4f}%")
    print(f"Observed policy difference: {result['top10_difference_percentage_points']:+.4f} pp")
    print('No policy-difference significance or production ROI is established.')


if __name__ == '__main__':
    main()
