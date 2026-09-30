"""Compatibility entry point for the canonical Digital Forge W4A8 converter."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "digital-forge-recipe" / "replicate_df_conversion.py"


if __name__ == "__main__":
    runpy.run_path(str(CONVERTER), run_name="__main__")
