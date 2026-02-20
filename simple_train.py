"""Deprecated wrapper for backward compatibility.

Use: python simple_main.py train
"""

import sys
from simple_main import main


if __name__ == "__main__":
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].startswith("-")):
        sys.argv = [sys.argv[0], "train", *sys.argv[1:]]
    raise SystemExit(main())
