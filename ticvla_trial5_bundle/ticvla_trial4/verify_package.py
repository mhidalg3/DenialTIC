#!/usr/bin/env python3
"""Verify ZIP payload survived transfer/unpacking unchanged (not model downloads)."""
from pathlib import Path
import hashlib
import json
import sys
root = Path(__file__).resolve().parent
expected = json.loads((root / "SHA256SUMS.json").read_text())
for relative, digest in expected.items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"Checksum mismatch or missing file: {relative}")
print(f"Trial 4 bootstrap package verified: {len(expected)} files")
