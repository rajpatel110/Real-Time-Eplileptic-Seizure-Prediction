import unittest
import numpy as np
from eeg_project.corrections import correct_rows


class CorrectionTests(unittest.TestCase):
    def test_exclusion_relabel_and_holdout_preservation(self):
        data = dict(X=np.arange(6).reshape(3, 2), y=np.array([0, 0, 1]),
                    recording=np.array(['dev','dev','test']), subject=np.array(['p']*3),
                    start_seconds=np.array([0,5,0]), end_seconds=np.array([10,15,10]))
        changes = [dict(recording='dev',start_seconds=0,end_seconds=10,old_label=0,proposed_label=None),
                   dict(recording='dev',start_seconds=5,end_seconds=15,old_label=0,proposed_label=1)]
        result = correct_rows(data,changes,{'dev'})
        np.testing.assert_array_equal(result['y'],[1,1])
        np.testing.assert_array_equal(result['X'][-1],data['X'][-1])
        np.testing.assert_array_equal(data['y'],[0,0,1])
        with self.assertRaises(ValueError):
            correct_rows(data,changes,{'test'})
        with self.assertRaises(ValueError):
            correct_rows(data,changes+changes,{'dev'})
        changes[0]['old_label']=1
        with self.assertRaises(ValueError):
            correct_rows(data,changes,{'dev'})
