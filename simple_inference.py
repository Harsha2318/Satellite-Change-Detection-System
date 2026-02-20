"""Deprecated wrapper for backward compatibility.

Use: python simple_main.py infer
"""

import sys
from simple_main import main


if __name__ == "__main__":
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].startswith("-")):
        sys.argv = [sys.argv[0], "infer", *sys.argv[1:]]
    elif len(sys.argv) > 1 and sys.argv[1] == "batch":
        sys.argv = [sys.argv[0], "infer"] + sys.argv[2:]
    raise SystemExit(main())
