"""Rebuild the production UI and fail when tracked assets were stale."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "chessy" / "api" / "static"


def snapshot() -> dict[str, str]:
    return {
        path.relative_to(STATIC).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in STATIC.rglob("*")
        if path.is_file()
    }


def main() -> int:
    before = snapshot()
    subprocess.run(["npm", "--prefix", "ui", "run", "build"], cwd=ROOT, check=True)
    after = snapshot()
    if before != after:
        raise SystemExit("production UI assets were stale; commit the rebuilt static directory")
    print(f"production UI is current ({len(after)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
