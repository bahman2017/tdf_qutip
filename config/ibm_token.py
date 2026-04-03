"""
Load IBM Quantum credentials from a **local** JSON file (not committed).

Default path: ``<repo>/apikey.json`` (see root ``.gitignore``).

Supported shapes::

    {"token": "..."}
    {"apikey": "..."}
    "plain-token-string"

Optional keys: ``channel`` (default ``ibm_quantum``), ``instance`` (some accounts).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_TOKEN_KEYS = (
    "token",
    "apikey",
    "API_KEY",
    "QISKIT_IBM_TOKEN",
    "ibm_quantum_token",
    "IAM_API_KEY",
)


def default_apikey_path() -> Path:
    return Path(__file__).resolve().parent.parent / "apikey.json"


def load_ibm_apikey_dict(path: Path | None = None) -> dict[str, Any] | None:
    """
    Read and parse ``apikey.json``. Returns ``None`` if the file is missing.
    """
    p = path or default_apikey_path()
    if not p.is_file():
        return None
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    data = json.loads(raw)
    if isinstance(data, str):
        return {"token": data.strip()}
    if isinstance(data, dict):
        return data
    raise ValueError(f"apikey.json must be a JSON object or string, got {type(data)}")


def extract_token(data: dict[str, Any]) -> str | None:
    for k in _TOKEN_KEYS:
        v = data.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def load_ibm_token(path: Path | None = None) -> str | None:
    """Return the API token string, or ``None`` if no file / no recognized key."""
    d = load_ibm_apikey_dict(path)
    if not d:
        return None
    return extract_token(d)


def ibm_runtime_channel(data: dict[str, Any] | None) -> str:
    # Runtime SDK accepts e.g. 'ibm_quantum_platform' or 'ibm_cloud' (not legacy 'ibm_quantum').
    if not data:
        return "ibm_quantum_platform"
    ch = data.get("channel")
    return str(ch).strip() if ch else "ibm_quantum_platform"


def _normalize_channel(ch: str) -> str:
    legacy = {"ibm_quantum": "ibm_quantum_platform"}
    return legacy.get(ch, ch)


def qiskit_runtime_service(apikey_path: Path | None = None):
    """
    ``QiskitRuntimeService`` using ``apikey.json`` when present, otherwise the
    default saved account / environment (IBM SDK behavior).
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    data = load_ibm_apikey_dict(apikey_path)
    token = extract_token(data) if data else None
    if token:
        ch = _normalize_channel(ibm_runtime_channel(data))
        kwargs: dict[str, str] = {
            "channel": ch,
            "token": token,
        }
        if data and data.get("instance"):
            kwargs["instance"] = str(data["instance"]).strip()
        return QiskitRuntimeService(**kwargs)
    return QiskitRuntimeService()
