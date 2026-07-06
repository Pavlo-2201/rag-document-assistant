"""Скопированный пакет semsearch не изменён относительно проверенной версии."""

import hashlib
import re
from pathlib import Path

VENDORED = Path(__file__).resolve().parents[1] / "src" / "semsearch"


def test_vendored_files_match_recorded_hashes():
    listing = (VENDORED / "VENDORED.md").read_text(encoding="utf-8")
    recorded = dict((name, digest) for digest, name in re.findall(r"^([0-9a-f]{64})  (\S+)$", listing, re.M))
    assert len(recorded) == 12
    for name, digest in recorded.items():
        assert hashlib.sha256((VENDORED / name).read_bytes()).hexdigest() == digest, name
