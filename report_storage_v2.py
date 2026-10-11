from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any


META_PREFIX = "<!-- report-meta:"
META_SUFFIX = "-->"


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def safe_entry_key(value: Any) -> str:
    text = str(value or "").strip()
    return "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in text)


def make_unique_entry_key(
    base_key: str,
    existing_keys: set[str] | list[str] | tuple[str, ...],
    *,
    now: dt.datetime | None = None,
) -> str:
    """Return base_key when unused, otherwise append timestamp to avoid overwrite."""
    base = safe_entry_key(base_key)
    existing = {safe_entry_key(x) for x in existing_keys}
    if base not in existing:
        return base

    current = now or dt.datetime.now()
    stamp = current.strftime("%Y%m%d_%H%M%S")
    candidate = f"{base}_{stamp}"
    seq = 2
    while candidate in existing:
        candidate = f"{base}_{stamp}_{seq}"
        seq += 1
    return candidate


def wrap_report_content(
    content: str,
    *,
    kind: str,
    key: str,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> str:
    created = created_at or now_iso()
    updated = updated_at or created
    meta = {
        "version": 2,
        "kind": str(kind or ""),
        "key": str(key or ""),
        "created_at": created,
        "updated_at": updated,
    }
    body = str(content or "")
    return (
        f"{META_PREFIX} {json.dumps(meta, ensure_ascii=False)} {META_SUFFIX}\n\n"
        f"{body}"
    )


def unwrap_report_content(content: str) -> tuple[str, dict[str, Any]]:
    text = str(content or "")
    first, sep, rest = text.partition("\n")
    if first.startswith(META_PREFIX) and first.rstrip().endswith(META_SUFFIX):
        raw = first[len(META_PREFIX):]
        raw = raw[: raw.rfind(META_SUFFIX)].strip()
        try:
            meta = json.loads(raw)
            if isinstance(meta, dict):
                return (rest.lstrip("\n") if sep else "", meta)
        except Exception:
            pass
    return text, {}


def display_entry_key(key: Any, kind: str = "") -> str:
    text = str(key or "")
    if kind == "analysis":
        m = re.match(r"^(\d{4}-\d{2}-\d{2})(?:_\d{8}_\d{6}(?:_\d+)?)?$", text)
        if m:
            return m.group(1)
    if kind == "earnings":
        m = re.match(r"^(.+?)(?:_\d{8}_\d{6}(?:_\d+)?)?$", text)
        if m:
            return m.group(1)
    return text


def merge_report_entries(
    local_entries: list[dict[str, Any]] | None,
    remote_entries: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """
    Merge local and remote entries by key.

    Remote wins when both exist, but local-only items stay visible and are marked
    LOCAL ONLY so a GitHub write failure never makes the report appear lost.
    """
    merged: dict[str, dict[str, Any]] = {}

    for item in local_entries or []:
        key = str(item.get("key", "") or "")
        if not key:
            continue
        row = dict(item)
        row["_sync_status"] = row.get("_sync_status") or "LOCAL ONLY"
        merged[key] = row

    for item in remote_entries or []:
        key = str(item.get("key", "") or "")
        if not key:
            continue
        row = dict(item)
        row["_sync_status"] = "GITHUB"
        merged[key] = row

    return sorted(
        merged.values(),
        key=lambda e: (
            str(e.get("updated_at", "") or ""),
            str(e.get("key", "") or ""),
        ),
        reverse=True,
    )
