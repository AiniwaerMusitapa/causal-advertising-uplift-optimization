"""Read-only environment and local-artifact checks; never train or deserialize models."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def check_dependencies(root=ROOT):
    problems = []
    if sys.version_info[:2] != (3, 11):
        problems.append('Use Python 3.11 to match the recorded experiment environment.')
    for line in (root / 'requirements.txt').read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s]+)', line)
        if not match:
            problems.append(f'Unsupported requirement pin: {line}')
            continue
        name, expected = match.groups()
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f'{name}: missing (expected {expected})')
            continue
        if actual != expected:
            problems.append(f'{name}: installed {actual}, expected {expected}')
    return problems


def verify_file(root, relative, record):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        return [f'{relative}: resolves outside this checkout; not read']
    if not path.is_file():
        return [f'{relative}: missing; regenerate with the README pipeline']
    expected_size = record['bytes']
    expected_hash = record['sha256']
    if (not isinstance(expected_size, int) or expected_size <= 0
            or not re.fullmatch(r'[0-9a-f]{64}', expected_hash)):
        return [f'{relative}: invalid size/checksum metadata']
    if path.stat().st_size != expected_size:
        return [f'{relative}: size mismatch; expected {expected_size} bytes']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected_hash:
        return [f'{relative}: SHA256 mismatch']
    return []


def check_artifacts(root=ROOT):
    def report(name):
        return json.loads((root / 'reports' / name).read_text(encoding='utf-8'))

    source = report('data_source.json')
    metrics = report('uplift_metrics.json')
    manifest = json.loads((root / 'experiment_manifest.json').read_text(encoding='utf-8'))
    problems = []
    if source['status'] != 'verified' or metrics['status'] != 'complete' or manifest['status'] != 'complete':
        problems.append('Source, model results and manifest must have completed statuses.')
    if source['sha256'] != manifest['source']['sha256']:
        problems.append('Source report and manifest disagree on the archive checksum.')
    files = [
        ('data/raw/criteo-research-uplift-v2.1.csv.gz', source),
        ('data/processed/criteo_uplift_v2_1.parquet', manifest['dataset']),
    ]
    if not metrics['model_artifacts']:
        problems.append('No model artifacts are recorded.')
    for name, record in metrics['model_artifacts'].items():
        if not re.fullmatch(r'[A-Za-z0-9_]+', name):
            problems.append(f'Invalid model name: {name}; not read')
            continue
        files.append((f'models/{name}.joblib', record))
        if manifest['model_artifacts'].get(name) != record:
            problems.append(f'{name}: model provenance differs between reports and manifest.')
    for relative, record in files:
        problems.extend(verify_file(root, relative, record))
    return problems, len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependencies', action='store_true', help='Check Python and installed pinned package versions')
    parser.add_argument('--artifacts', action='store_true', help='Hash the local archive, Parquet and recorded model files')
    args = parser.parse_args()
    if not (args.dependencies or args.artifacts):
        parser.error('Choose --dependencies, --artifacts, or both. No download or training is performed.')
    problems = []
    try:
        if args.dependencies:
            found = check_dependencies()
            problems.extend(found)
            print(f'Dependency versions: {"FAIL" if found else "PASS"}')
        if args.artifacts:
            found, count = check_artifacts()
            problems.extend(found)
            print(f'Local file integrity ({count} files): {"FAIL" if found else "PASS"}')
    except (OSError, ValueError, KeyError, TypeError) as error:
        problems.append(f'Unable to complete checks: {error}')
    for problem in problems:
        print(f'- {problem}')
    print('These checks do not establish numerical reproducibility or causal validity.')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
