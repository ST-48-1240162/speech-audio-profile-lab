import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestAblation(unittest.TestCase):
    def test_mock_ablation_generates_table(self):
        out = ROOT / "results" / "test-ablation"
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_ablation.py"),
                "--mock",
                "--max-utt",
                "4",
                "--out",
                str(out),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue((out / "ABLATION.md").is_file())
        self.assertTrue((out / "ablation_table.csv").is_file())
        text = (out / "ABLATION.md").read_text(encoding="utf-8")
        self.assertIn("wav2vec2-heads", text)
        self.assertIn("telephony", text)


if __name__ == "__main__":
    unittest.main()
