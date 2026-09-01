#!/usr/bin/env python3
"""Shared manifest and run-contract checks for counterfactual collection."""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


RUN_ROUTE_FIELDS = (
    "route_id",
    "town",
    "source_route_id",
    "scenario_name",
    "scenario_type",
)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(value: Any) -> str | None:
    return None if value is None else str(value)


def route_xml_contract(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        root = ET.parse(path).getroot()
    except (FileNotFoundError, ET.ParseError, OSError) as exc:
        return None, [f"unreadable route XML {path}: {exc}"]
    routes = root.findall("route")
    scenarios = root.findall(".//scenario")
    if len(routes) != 1:
        errors.append(f"route XML must contain exactly one route, found {len(routes)}")
    if len(scenarios) != 1:
        errors.append(f"route XML must contain exactly one scenario, found {len(scenarios)}")
    if errors:
        return None, errors
    route = routes[0]
    scenario = scenarios[0]
    contract = {
        "route_id": route.attrib.get("id"),
        "town": route.attrib.get("town"),
        "source_route_id": route.attrib.get("source_route_id"),
        "scenario_name": scenario.attrib.get("name"),
        "scenario_type": scenario.attrib.get("type"),
        "route_sha256": sha256(path),
    }
    for field in ("route_id", "town", "scenario_name", "scenario_type"):
        if not contract.get(field):
            errors.append(f"route XML is missing {field}")
    return contract, errors


def validate_manifest_job(job: dict[str, Any]) -> list[str]:
    """Check that a manifest row still describes the current frozen XML."""
    errors: list[str] = []
    required = (
        "schema_version",
        "job_id",
        "group_id",
        "scenario_id",
        "decision",
        "route_path",
        "route_sha256",
        "route_id",
        "town",
        "scenario_name",
        "scenario_type",
        "output_dir",
    )
    for field in required:
        if job.get(field) in (None, ""):
            errors.append(f"manifest row is missing {field}")
    route_path = Path(str(job.get("route_path", "")))
    if not route_path.is_file():
        errors.append(f"manifest route does not exist: {route_path}")
        return errors
    contract, xml_errors = route_xml_contract(route_path)
    errors.extend(xml_errors)
    if contract is None:
        return errors
    for field in (*RUN_ROUTE_FIELDS, "route_sha256"):
        if normalized(job.get(field)) != normalized(contract.get(field)):
            errors.append(
                f"manifest {field}={job.get(field)!r} does not match route XML "
                f"{contract.get(field)!r}"
            )
    return errors


def validate_run_contract(run_dir: Path, job: dict[str, Any] | None = None) -> list[str]:
    """Bind a recorded branch to its copied XML and optional manifest row."""
    errors: list[str] = []
    spec_path = run_dir / "metadata/run_spec.json"
    source_path = run_dir / "metadata/source_route.xml"
    spec = read_json(spec_path, {}) or {}
    route = spec.get("route") or {}
    if not spec:
        errors.append("missing readable metadata/run_spec.json")
    if not source_path.is_file():
        errors.append("missing metadata/source_route.xml")
        source_contract = None
    else:
        source_contract, source_errors = route_xml_contract(source_path)
        errors.extend(f"source route: {error}" for error in source_errors)

    if source_contract is not None:
        for field in (*RUN_ROUTE_FIELDS, "route_sha256"):
            if normalized(route.get(field)) != normalized(source_contract.get(field)):
                errors.append(
                    f"run_spec route.{field}={route.get(field)!r} does not match copied "
                    f"source route {source_contract.get(field)!r}"
                )

    if job is not None:
        if normalized(spec.get("schema_version")) != normalized(job.get("schema_version")):
            errors.append(
                f"run schema {spec.get('schema_version')!r} does not match manifest "
                f"{job.get('schema_version')!r}"
            )
        if normalized(spec.get("decision")) != normalized(job.get("decision")):
            errors.append(
                f"run decision {spec.get('decision')!r} does not match manifest "
                f"{job.get('decision')!r}"
            )
        for field in (*RUN_ROUTE_FIELDS, "route_sha256"):
            if normalized(route.get(field)) != normalized(job.get(field)):
                errors.append(
                    f"run route.{field}={route.get(field)!r} does not match manifest "
                    f"{job.get(field)!r}"
                )
    return errors


def finite_point(element: ET.Element | None) -> bool:
    if element is None:
        return False
    try:
        return all(math.isfinite(float(element.attrib[key])) for key in ("x", "y", "z"))
    except (KeyError, TypeError, ValueError):
        return False
