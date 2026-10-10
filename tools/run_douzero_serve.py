import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "DouZero"))
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tools.douzero_serve import serve

if __name__ == "__main__":
    serve(18765, str(ROOT / "baselines" / "douzero_ADP"))
