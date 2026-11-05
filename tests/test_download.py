"""Exercise cache reuse and failed transfers without accessing the internet."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from eeg_project.dataset import download_sample


class DownloadTests(unittest.TestCase):
    def test_verified_transfer_then_cached_rerun(self):
        payload = b"test recording bytes"
        checksum = hashlib.sha256(payload).hexdigest()
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [payload[:5], b"", payload[5:]]
        with tempfile.TemporaryDirectory() as directory:
            with patch("eeg_project.dataset.SAMPLE_FILES", {"chb01/test.edf": checksum}), patch("eeg_project.dataset.requests.get", return_value=response) as network:
                paths = download_sample(Path(directory))
                self.assertEqual(paths[0].read_bytes(), payload)
                self.assertFalse(paths[0].with_suffix(".edf.part").exists())
                download_sample(Path(directory))
                self.assertEqual(network.call_count, 1)

    def test_bad_transfer_never_becomes_final_file(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [b"bad bytes"]
        with tempfile.TemporaryDirectory() as directory:
            with patch("eeg_project.dataset.SAMPLE_FILES", {"chb01/test.edf": "0" * 64}), patch("eeg_project.dataset.requests.get", return_value=response):
                with self.assertRaises(ValueError):
                    download_sample(Path(directory))
                self.assertFalse((Path(directory) / "chb01/test.edf").exists())
                self.assertTrue((Path(directory) / "chb01/test.edf.part").exists())

    def test_network_failure_does_not_create_final_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("eeg_project.dataset.SAMPLE_FILES", {"chb01/test.edf": "0" * 64}), patch("eeg_project.dataset.requests.get", side_effect=ConnectionError("offline")):
                with self.assertRaises(ConnectionError):
                    download_sample(Path(directory))
                self.assertFalse((Path(directory) / "chb01/test.edf").exists())
