"""
pytest configuration: add 'src/' to sys.path so that internal imports
inside the src package (e.g. ``from config import …``) work correctly
when tests import modules via ``from src.xxx import …``.
"""

import sys
from pathlib import Path

# Allow ``from config import …`` style imports used inside src/*.py
sys.path.insert(0, str(Path(__file__).parent / "src"))
