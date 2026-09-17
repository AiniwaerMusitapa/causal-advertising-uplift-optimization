"""Exercise preflight failures without installing ML packages or reading real data."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('preflight', ROOT / 'scripts/06_preflight.py')
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class PreflightChecks(unittest.TestCase):
    def test_valid_corrupt_truncated_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'fixture.bin'
            data = b'known test bytes'
            record = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            path.write_bytes(data)
            self.assertEqual(PREFLIGHT.verify_file(root, 'fixture.bin', record), [])
            path.write_bytes(b'x' * len(data))
            self.assertIn('SHA256 mismatch', PREFLIGHT.verify_file(root, 'fixture.bin', record)[0])
            path.write_bytes(b'x')
            self.assertIn('size mismatch', PREFLIGHT.verify_file(root, 'fixture.bin', record)[0])
            self.assertIn('missing', PREFLIGHT.verify_file(root, 'absent.bin', record)[0])

    def test_paths_outside_checkout_are_not_read(self):
        with tempfile.TemporaryDirectory() as directory:
            result = PREFLIGHT.verify_file(Path(directory), '../external.bin', {})
            self.assertIn('outside this checkout; not read', result[0])

    def test_invalid_record_does_not_claim_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'fixture.bin').write_bytes(b'x')
            result = PREFLIGHT.verify_file(root, 'fixture.bin', {'bytes': 1, 'sha256': 'not-a-hash'})
            self.assertIn('invalid size/checksum metadata', result[0])

    def test_dependency_versions_and_missing_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'requirements.txt').write_text('numpy==1.26.4\npandas==2.2.3\n', encoding='utf-8')
            with mock.patch.object(PREFLIGHT.sys, 'version_info', (3, 11, 9)):
                with mock.patch.object(PREFLIGHT.metadata, 'version', side_effect=['1.26.4', '2.2.3']):
                    self.assertEqual(PREFLIGHT.check_dependencies(root), [])
                with mock.patch.object(PREFLIGHT.metadata, 'version', side_effect=['0.0', PREFLIGHT.metadata.PackageNotFoundError('pandas')]):
                    result = PREFLIGHT.check_dependencies(root)
                    self.assertEqual(len(result), 2)
                    self.assertIn('installed 0.0', result[0])
                    self.assertIn('missing', result[1])

    def test_missing_reports_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                PREFLIGHT.check_artifacts(Path(directory))

    def test_cli_requires_explicit_scope(self):
        process = subprocess.run([sys.executable, str(ROOT / 'scripts/06_preflight.py')], capture_output=True, text=True)
        self.assertEqual(process.returncode, 2)
        self.assertIn('Choose --dependencies', process.stderr)
