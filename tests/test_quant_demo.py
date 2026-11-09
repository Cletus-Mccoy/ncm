"""The demo at the bottom of models/quant.py must run (it builds a small quantizer, encodes and decodes one input).

Run: python tests/test_quant_demo.py   (or pytest tests/test_quant_demo.py)
"""
import os, subprocess, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def test_quant_main_demo_runs():
    r = subprocess.run([sys.executable, "-m", "models.quant"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    assert "torch.Size([1, 112])" in r.stdout          # 16 + 32 + 64 codes
    assert "torch.Size([1, 64, 256])" in r.stdout       # reconstruction


if __name__ == "__main__":
    test_quant_main_demo_runs()
    print("ok")
