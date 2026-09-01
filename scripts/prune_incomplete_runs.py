#!/usr/bin/env python3
"""Archive diagnostics and optionally remove rejected retry sensor payloads."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


RL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = RL_ROOT / "data/counterfactual_decision_v1"


def read_text(path: Path, limit: int = 16_384) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    if root != DEFAULT_ROOT.resolve():
        raise SystemExit(f"refusing non-campaign root: {root}")

    candidates = sorted(
        path for path in root.rglob("*_incomplete_*")
        if path.is_dir() and not path.is_symlink()
    )
    rows = []
    for path in candidates:
        resolved = path.resolve()
        if root not in resolved.parents or "_incomplete_" not in path.name:
            raise RuntimeError(f"unsafe candidate: {path}")
        quality_path = path / "quality_report.json"
        try:
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            quality = None
        size_bytes = sum(
            item.stat().st_size
            for item in path.rglob("*")
            if item.is_file() and not item.is_symlink()
        )
        rows.append(
            {
                "path": str(resolved),
                "size_bytes": size_bytes,
                "quality_report": quality,
                "run_env_tail": read_text(path / "run_env.txt"),
                "scenario_validation_log": read_text(path / "logs/scenario_validation.log"),
                "branch_console_tail": read_text(path / "logs/branch_console.log"),
            }
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = root / "_rejected_diagnostics"
    archive_dir.mkdir(parents=True, exist_ok=True)
    report_path = archive_dir / f"incomplete_runs_{stamp}.json"
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "delete_requested": args.delete,
        "directory_count": len(rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "runs": rows,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.delete:
        for path in candidates:
            shutil.rmtree(path)

    print(json.dumps({
        "report": str(report_path),
        "directories": len(rows),
        "bytes": report["total_size_bytes"],
        "deleted": args.delete,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
