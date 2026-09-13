import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_ml_jd_lab.sklearn_baseline import train_eval_baseline


class TestSklearnBaseline(unittest.TestCase):
    def test_sklearn_baseline_meets_synthetic_threshold(self):
        report = train_eval_baseline()
        self.assertGreaterEqual(report.accuracy, 0.85)
        self.assertGreaterEqual(report.f1, 0.85)
        self.assertGreaterEqual(report.roc_auc, 0.90)
        self.assertEqual(report.n_train + report.n_test, 80)


if __name__ == "__main__":
    unittest.main()
