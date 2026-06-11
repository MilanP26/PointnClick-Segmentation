from __future__ import annotations

import sys
from pathlib import Path


repo_src = Path(__file__).resolve().parent / "src"
if repo_src.exists():
    sys.path.insert(0, str(repo_src))

from pointnclick_segmentation.bridge_app import main


if __name__ == "__main__":
    main()
