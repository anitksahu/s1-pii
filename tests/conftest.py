import sys
from pathlib import Path

# the v2 tests reuse fixtures from tests/test_model.py (``from tests.test_model import ...``)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
