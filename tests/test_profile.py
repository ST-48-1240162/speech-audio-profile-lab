import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_ml_jd_lab.models import HFEmbeddingResult
from audio_ml_jd_lab.backends import BACKEND_HEADS
from audio_ml_jd_lab.profile import SCHEMA_VERSION, build_profile, build_profile_from_heads


class TestProfile(unittest.TestCase):
    @patch("audio_ml_jd_lab.chain.wav2vec2_embedding")
    def test_build_profile_shape(self, fake_hf):
        fake_hf.return_value = HFEmbeddingResult(
            model_id="facebook/wav2vec2-base-960h",
            embedding_dim=768,
            mean_activation=0.1,
            std_activation=0.9,
        )
        audio = np.zeros(16_000, dtype=np.float32)
        profile = build_profile_from_heads(audio, sr=16_000, lane="clean")
        d = profile.to_dict()
        self.assertEqual(d["schema_version"], SCHEMA_VERSION)
        self.assertEqual(d["backend"], BACKEND_HEADS)
        self.assertAlmostEqual(d["duration_s"], 1.0)
        self.assertTrue(d["language"]["abstain"])
        self.assertTrue(d["asr"]["abstain"])
        self.assertIn("inference", d["latency_ms"])
        self.assertEqual(d["quality"]["embedding_dim"], 768)

    @patch("audio_ml_jd_lab.backends.whisper_transcribe")
    def test_build_profile_whisper(self, fake_whisper):
        from audio_ml_jd_lab.backends import ASRResult

        fake_whisper.return_value = (
            ASRResult("mock/whisper", "hello", None, None),
            10.0,
        )
        audio = np.zeros(16_000, dtype=np.float32)
        profile = build_profile(audio, backend="whisper-base", lane="telephony")
        d = profile.to_dict()
        self.assertEqual(d["backend"], "whisper-base")
        self.assertEqual(d["lane"], "telephony")
        self.assertFalse(d["asr"]["abstain"])


if __name__ == "__main__":
    unittest.main()
