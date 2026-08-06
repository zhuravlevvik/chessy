from __future__ import annotations
import re
from datetime import datetime, timezone
_RUN_RE=re.compile(r"^\d{8}-\d{6}-[a-z0-9]+(?:-[a-z0-9]+)*-[0-9a-f]{10}(?:-\d+)?$")
def slugify(name: str) -> str:
    value=re.sub(r"[^a-z0-9]+","-",name.lower()).strip("-")
    return value[:40].strip("-") or "run"
def make_run_id(name: str, fingerprint: str, now: datetime | None=None) -> str:
    now=now or datetime.now(timezone.utc); return f"{now.strftime('%Y%m%d-%H%M%S')}-{slugify(name)}-{fingerprint[:10]}"
def valid_run_id(value: str) -> bool: return _RUN_RE.fullmatch(value) is not None
