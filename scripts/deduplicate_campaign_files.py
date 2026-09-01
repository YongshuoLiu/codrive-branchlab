#!/usr/bin/env python3
"""Losslessly hard-link byte-identical campaign files to conserve disk space."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


RL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = RL_ROOT / "data/counterfactual_decision_v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--minimum-bytes", type=int, default=64 * 1024)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    if root != DEFAULT_ROOT.resolve():
        raise SystemExit(f"refusing non-campaign root: {root}")

    by_size_mode: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        stat = path.stat()
        if stat.st_size >= args.minimum_bytes:
            by_size_mode[(stat.st_size, stat.st_mode)].append(path)

    duplicate_files = 0
    reclaimable_bytes = 0
    linked_files = 0
    for (size, _mode), paths in by_size_mode.items():
        if len(paths) < 2:
            continue
        by_hash: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            by_hash[digest(path)].append(path)
        for matches in by_hash.values():
            if len(matches) < 2:
                continue
            canonical = matches[0]
            canonical_inode = canonical.stat().st_ino
            for duplicate in matches[1:]:
                if duplicate.stat().st_ino == canonical_inode:
                    continue
                duplicate_files += 1
                reclaimable_bytes += size
                if not args.apply:
                    continue
                temporary = duplicate.with_name(f".{duplicate.name}.dedupe-{os.getpid()}")
                if temporary.exists():
                    temporary.unlink()
                os.link(canonical, temporary)
                # Re-check immediately before the atomic replacement. This
                # script is only run while collection is paused.
                if digest(duplicate) != digest(canonical):
                    temporary.unlink()
                    raise RuntimeError(f"file changed during deduplication: {duplicate}")
                os.replace(temporary, duplicate)
                linked_files += 1

    report = {
        "root": str(root),
        "minimum_bytes": args.minimum_bytes,
        "apply": args.apply,
        "duplicate_files": duplicate_files,
        "linked_files": linked_files,
        "reclaimable_bytes": reclaimable_bytes,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
