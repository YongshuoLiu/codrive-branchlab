#!/usr/bin/env python3
"""Inventory and remove derived augmented views while retaining raw modalities."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


RL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = RL_ROOT / "data/counterfactual_decision_v1"
AUGMENTED_NAMES = {
    "rgb_augmented",
    "depth_augmented",
    "semantics_augmented",
    "bev_semantics_augmented",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if root != DEFAULT_ROOT.resolve():
        raise SystemExit(f"refusing non-campaign root: {root}")

    directories = sorted(
        path
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink() and path.name in AUGMENTED_NAMES
    )
    rows = []
    for directory in directories:
        resolved = directory.resolve()
        if root not in resolved.parents or directory.name not in AUGMENTED_NAMES:
            raise RuntimeError(f"unsafe augmented directory: {directory}")
        files = [path for path in directory.rglob("*") if path.is_file() and not path.is_symlink()]
        rows.append(
            {
                "path": str(resolved),
                "modality": directory.name,
                "file_count": len(files),
                "bytes": sum(path.stat().st_size for path in files),
                "extensions": dict(sorted(Counter(path.suffix for path in files).items())),
            }
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = root / "_storage_diagnostics"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"pruned_augmented_modalities_{stamp}.json"
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "apply": args.apply,
        "directory_count": len(rows),
        "file_count": sum(row["file_count"] for row in rows),
        "bytes": sum(row["bytes"] for row in rows),
        "retained_modalities": [
            "rgb",
            "depth",
            "semantics",
            "bev_semantics",
            "lidar",
            "boxes",
            "measurements",
            "carla_telemetry_20hz",
            "decision_events",
            "collision_events",
        ],
        "directories": rows,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.apply:
        for directory in directories:
            shutil.rmtree(directory)
    print(json.dumps({
        "report": str(report_path),
        "directories": len(rows),
        "files": report["file_count"],
        "bytes": report["bytes"],
        "removed": args.apply,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
