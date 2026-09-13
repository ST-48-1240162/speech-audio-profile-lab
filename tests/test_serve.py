import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient

from audio_ml_jd_lab.models import HFEmbeddingResult
from audio_ml_jd_lab.serve import app


class TestServe(unittest.TestCase):
    @patch("audio_ml_jd_lab.chain.wav2vec2_embedding")
    def test_health_and_demo(self, fake_hf):
        fake_hf.return_value = HFEmbeddingResult(
            model_id="facebook/wav2vec2-base-960h",
            embedding_dim=768,
            mean_activation=0.0,
            std_activation=1.0,
        )
        client = TestClient(app)
        self.assertEqual(client.get("/health").json()["status"], "ok")
        demo = client.get("/demo").json()
        self.assertIn("sklearn", demo)
        profile = client.get("/demo/profile").json()
        self.assertEqual(profile["schema_version"], "1.0")
        self.assertEqual(profile["backend"], "wav2vec2-heads")
        self.assertTrue(profile["language"]["abstain"])
        metrics = client.get("/eval/sklearn-baseline").json()
        self.assertGreaterEqual(metrics["accuracy"], 0.85)


if __name__ == "__main__":
    unittest.main()
