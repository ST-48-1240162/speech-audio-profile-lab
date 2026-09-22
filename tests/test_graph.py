import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_ml_jd_lab.graph import build_profile_agent_graph


class TestLangGraphAgent(unittest.TestCase):
    def test_langgraph_state_graph_compiles(self):
        app = build_profile_agent_graph()
        self.assertIsNotNone(app)
        # Four-node pipeline: preprocess → infer → validate → package
        self.assertTrue(hasattr(app, "invoke"))


if __name__ == "__main__":
    unittest.main()
