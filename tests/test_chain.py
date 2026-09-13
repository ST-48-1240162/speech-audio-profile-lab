import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_ml_jd_lab.audio_io import synth_window
from audio_ml_jd_lab.chain import run_chain
from audio_ml_jd_lab.models import HFEmbeddingResult


class TestChain(unittest.TestCase):
    @patch("audio_ml_jd_lab.chain.wav2vec2_embedding")
    def test_langchain_pipeline_orchestrates_steps(self, fake_hf):
        fake_hf.return_value = HFEmbeddingResult(
            model_id="facebook/wav2vec2-base-960h",
            embedding_dim=768,
            mean_activation=0.01,
            std_activation=0.5,
        )
        audio = synth_window(freq_hz=150.0, seed=3)
        out = run_chain(audio)
        self.assertIn("sklearn", out)
        self.assertEqual(out["huggingface"]["embedding_dim"], 768)
        self.assertEqual(out["mfcc_dim"], 26)


if __name__ == "__main__":
    unittest.main()
