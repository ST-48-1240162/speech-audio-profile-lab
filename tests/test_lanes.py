import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_ml_jd_lab.audio_io import synth_window
from audio_ml_jd_lab.lanes import apply_lane


class TestLanes(unittest.TestCase):
    def test_clean_passthrough(self):
        audio = synth_window(seed=1)
        out = apply_lane(audio, sr=16_000, lane="clean")
        np.testing.assert_allclose(out, audio)

    def test_telephony_changes_signal(self):
        audio = synth_window(seed=2)
        out = apply_lane(audio, sr=16_000, lane="telephony")
        self.assertFalse(np.allclose(out, audio))

    def test_noise_telecom_differs_from_telephony(self):
        audio = synth_window(seed=3)
        tel = apply_lane(audio, sr=16_000, lane="telephony")
        noisy = apply_lane(audio, sr=16_000, lane="noise_telecom")
        self.assertFalse(np.allclose(tel, noisy))


if __name__ == "__main__":
    unittest.main()
