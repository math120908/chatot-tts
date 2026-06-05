#!/usr/bin/env python3
"""CLI script to migrate chatot config.yaml from v1 to v2 format.

Usage:
    python scripts/migrate_config.py              # migrate default config
    python scripts/migrate_config.py path/to/config.yaml

This script can be deleted once all configs have been migrated.
"""

from __future__ import annotations

import sys
from pathlib import Path

from chatot._migrate import migrate_file

_DEFAULT_CONFIG_PATH = Path("~/.config/chatot/config.yaml").expanduser()


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).expanduser()
    else:
        path = _DEFAULT_CONFIG_PATH

    if not path.exists():
        print(f"Config file not found: {path}")
        sys.exit(1)

    if migrate_file(path):
        print(f"Migrated: {path}")
    else:
        print(f"Already v2 format (no changes): {path}")


if __name__ == "__main__":
    main()
