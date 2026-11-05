"""Synthetic signal, leakage, evaluation and runner regression tests; no EDF/network."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from eeg_project.dataset import Seizure, download_sample
from eeg_project.evaluation import binary_metrics, select_threshold, split_masks
from eeg_project.experiment import run
from eeg_project.features import extract_features, filter_signal, window_label, preprocess, process
from eeg_project.intensity import window_amplitude_uv, relative_amplitude


class PipelineTests(unittest.TestCase):
    def test_band_features_and_amplitude(self):
        t = np.arange(2560) / 256
        signal = (2e-6 * np.sin(2 * np.pi * 10 * t))[None, :]
        values = extract_features(signal, 256).reshape(1, 7)
        self.assertEqual(values.shape, (1, 7))
        self.assertGreater(values[0, 5], 100 * values[0, 4])
        names = ['x:' + n for n in ['mean', 'std', 'variance', 'delta', 'theta', 'alpha', 'beta']]
        np.testing.assert_allclose(window_amplitude_uv(values, names), [np.sqrt(2)])
        np.testing.assert_allclose(relative_amplitude([2, 4], 2), [1, 2])

    def test_causal_filter(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(size=(2, 1024))
        changed = signal.copy()
        changed[:, 512:] += 100
        filters = dict(low_hz=0.5, high_hz=40, notch_hz=50)
        before = filter_signal(signal, 256, filters)
        after = filter_signal(changed, 256, filters)
        np.testing.assert_array_equal(before[:, :512], after[:, :512])
        with self.assertRaises(ValueError):
            filter_signal(signal, 80, filters)
        with self.assertRaises(ValueError):
            extract_features([[np.nan] * 256], 256)

    def test_edf_resource_limit_precedes_allocation(self):
        raw = MagicMock()
        raw.__enter__.return_value = raw
        raw.info = {"sfreq": 256}
        raw.ch_names = ["x"]
        raw.n_times = 50_000_001
        with patch("eeg_project.features.read_edf_channel_labels", return_value=["x"]), patch("eeg_project.features.mne.io.read_raw_edf", return_value=raw):
            with self.assertRaisesRegex(ValueError, "sample limit"):
                preprocess(Path("synthetic.edf"), {})
        raw.get_data.assert_not_called()

    def test_window_configuration_errors(self):
        settings = dict(window_seconds=10, stride_seconds=0, preictal_seconds=600, postictal_seconds=600)
        with patch("eeg_project.features.preprocess", return_value=(("x",), 256, np.zeros((1, 2560)))):
            with self.assertRaisesRegex(ValueError, "stride"):
                process(Path("synthetic.edf"), [], {}, settings)
            settings["stride_seconds"] = 5
            with self.assertRaisesRegex(ValueError, "annotation"):
                process(Path("synthetic.edf"), [Seizure(9, 11)], {}, settings)

    def test_label_boundaries(self):
        events = [Seizure(100, 110)]
        for start, end, expected in [(0, 10, 0), (75, 85, None), (80, 90, 1),
                                     (90, 100, 1), (95, 105, None), (110, 120, None), (130, 140, 0)]:
            with self.subTest(start=start):
                self.assertEqual(window_label(start, end, events, 20, 20), expected)
        self.assertIsNone(window_label(120, 130, [Seizure(100, 110), Seizure(140, 150)], 20, 20))

    def test_metrics(self):
        metrics = binary_metrics([0, 0, 1, 1], [.1, .8, .3, .9], .5)
        self.assertEqual(metrics['confusion_matrix'], [[1, 1], [1, 1]])
        self.assertEqual(metrics['precision'], .5)
        self.assertEqual(metrics['recall'], .5)
        self.assertEqual(metrics['f1'], .5)
        self.assertAlmostEqual(metrics['average_precision'], 5/6)
        self.assertIsNone(binary_metrics([0, 0], [.1, .2], .5)['average_precision'])
        self.assertIsNone(select_threshold([1, 0], [.1, .9], 1, 1))
        with self.assertRaises(ValueError):
            binary_metrics([0, 1], [0, np.nan], .5)

    def test_split_leakage_guards(self):
        splits = dict(train=['a'], validation=['b'], test=['c'])
        masks = split_masks(['a', 'a', 'b', 'c'], ['p', 'p', 'p', 'p'], splits)
        self.assertEqual(int(masks['train'].sum()), 2)
        with self.assertRaisesRegex(ValueError, 'Patient leakage'):
            split_masks(['a', 'b', 'c'], ['p'] * 3, splits, True)
        for bad in [dict(train=['a'], validation=['a'], test=['c']),
                    dict(train=['a'], validation=['b'], test=['missing'])]:
            with self.assertRaises(ValueError):
                split_masks(['a', 'b', 'c'], ['p'] * 3, bad)

    def test_download_paths(self):
        with tempfile.TemporaryDirectory() as directory, patch('eeg_project.dataset.requests.get') as network:
            for path in ['../outside.edf', '/absolute.edf', 'chb01/../../x', 'chb01/x:stream', 'chb01\\x']:
                with self.assertRaises(ValueError):
                    download_sample(Path(directory), {path: '0' * 64})
            network.assert_not_called()

    def test_runner_reproducibility_and_holdout_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(42)
            X = rng.normal(size=(18, 7))
            y = np.tile([0, 1], 9)
            recording = np.repeat(['a', 'b', 'c'], 6)
            def save():
                np.savez(root/'data.npz', X=X, y=y, recording=recording,
                         subject=np.repeat(['p', 'q', 'r'], 6), feature_names=[f'x:{i}' for i in range(7)],
                         start_seconds=np.tile(np.arange(6)*5, 3), end_seconds=np.tile(np.arange(6)*5+10, 3),
                         configuration=json.dumps({'synthetic': True}))
            save()
            config = root/'run.toml'
            config.write_text('''schema_version = 1
archive = "data.npz"
seed = 42
patient_independent = true
[model]
transform = "signed_log"
C = 0.01
max_iter = 3000
[selection]
min_precision = 0.5
min_recall = 0.5
[splits]
train = ["a"]
validation = ["b"]
test = ["c"]
''')
            first = run(config, root/'first.json')
            second = run(config, root/'second.json')
            self.assertEqual(first, second)
            X[12:] = 1e30
            y[12:] = 0
            save()
            changed = run(config, root/'changed.json')
            self.assertEqual(first['validation_at_fixed_0_5'], changed['validation_at_fixed_0_5'])
            self.assertEqual(first['selected_threshold'], changed['selected_threshold'])
            with self.assertRaisesRegex(ValueError, 'already exists'):
                run(config, root/'first.json')


if __name__ == '__main__':
    unittest.main()
