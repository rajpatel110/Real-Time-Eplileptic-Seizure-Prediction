import unittest
from eeg_project.features import window_label
from eeg_project.timeline import summary_timeline


def record(name, start, end, event=''):
    return f'File Name: {name}\nFile Start Time: {start}\nFile End Time: {end}\nNumber of Seizures in File: {int(bool(event))}\n{event}'


class TimelineTests(unittest.TestCase):
    def test_extended_midnight_and_gap(self):
        text = record('chb01_01.edf', '23:30:00', '24:30:00') + record('chb01_02.edf', '00:30:08', '01:30:08')
        result = summary_timeline(text)
        self.assertEqual(result['chb01_02.edf'].start - result['chb01_01.edf'].end, 8)
        self.assertEqual(result['chb01_01.edf'].end - result['chb01_01.edf'].start, 3600)

    def test_cross_file_exclusion(self):
        text = record('chb01_01.edf', '10:00:00', '11:00:00', 'Seizure Start Time: 3500 seconds\nSeizure End Time: 3550 seconds\n')
        text += record('chb01_02.edf', '11:00:10', '12:00:10')
        result = summary_timeline(text)
        start = result['chb01_02.edf'].start
        self.assertIsNone(window_label(start, start+10, result['chb01_01.edf'].events, 600, 600))
        self.assertEqual(window_label(start+540, start+550, result['chb01_01.edf'].events, 600, 600), 0)

    def test_invalid_clock_overlap_and_event(self):
        cases = [record('chb01_01.edf','25:00:00','26:00:00'),
                 record('chb01_01.edf','10:00:00','11:00:00') + record('chb01_02.edf','10:30:00','11:30:00'),
                 record('chb01_01.edf','10:00:00','11:00:00','Seizure Start Time: 3500 seconds\nSeizure End Time: 3700 seconds\n')]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                summary_timeline(text)
