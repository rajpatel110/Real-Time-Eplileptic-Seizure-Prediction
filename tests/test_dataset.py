"""Annotation and download integrity checks using local fixtures, no network."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eeg_project.dataset import SAMPLE_FILES, download_sample, parse_summary


def section(events, count=1, filename="chb01_15.edf"):
    return f"File Name: {filename}\nFile Start Time: 24:44:29\nNumber of Seizures in File: {count}\n{events}"


class SummaryTests(unittest.TestCase):
    def test_single_event(self):
        record = parse_summary(section("Seizure Start Time: 1732 seconds\nSeizure End Time: 1772 seconds\n"))["chb01_15.edf"]
        self.assertEqual((record.seizures[0].start, record.seizures[0].end), (1732, 1772))

    def test_zero_and_multiple_events(self):
        text = section("", 0, "chb01_01.edf") + section(
            "Seizure 1 Start Time: 10 seconds\nSeizure 1 End Time: 20 seconds\n"
            "Seizure 2 Start Time: 30 seconds\nSeizure 2 End Time: 50 seconds\n", 2)
        result = parse_summary(text)
        self.assertEqual(result["chb01_01.edf"].seizures, ())
        self.assertEqual(len(result["chb01_15.edf"].seizures), 2)

    def test_invalid_summaries_fail_instead_of_becoming_negative_labels(self):
        cases = ["", "File Name: chb01_01.edf\n", section("", 1),
                 section("Seizure Start Time: 20 seconds\nSeizure End Time: 10 seconds\n"),
                 section("Seizure Start Time: 20 seconds\n"),
                 section("Seizure Start Time: nonsense seconds\n"),
                 section("Seizure 1 Start Time: 10 seconds\nSeizure 2 End Time: 20 seconds\n"),
                 section("", 0) + section("", 0),
                 section("Seizure 1 Start Time: 10 seconds\nSeizure 1 End Time: 30 seconds\n"
                         "Seizure 2 Start Time: 20 seconds\nSeizure 2 End Time: 40 seconds\n", 2)]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_summary(text)

    def test_corrupt_existing_download_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / next(iter(SAMPLE_FILES))
            target.parent.mkdir(parents=True)
            target.write_bytes(b"corrupt")
            with patch("eeg_project.dataset.requests.get") as network:
                with self.assertRaises(ValueError):
                    download_sample(Path(directory))
                network.assert_not_called()
            self.assertEqual(target.read_bytes(), b"corrupt")


if __name__ == "__main__":
    unittest.main()
