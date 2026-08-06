from __future__ import annotations

import re
from importlib.resources import files


def test_packaged_static_index_references_existing_hashed_assets() -> None:
    static = files("chessy.api").joinpath("static")
    index = static.joinpath("index.html").read_text(encoding="utf-8")
    references = re.findall(r'(?:src|href)="(/assets/[^"]+)"', index)
    assert references
    for reference in references:
        assert static.joinpath(reference.removeprefix("/")).is_file()
