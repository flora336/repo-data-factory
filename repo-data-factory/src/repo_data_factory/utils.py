from __future__ import annotations
import hashlib
import os
from pathlib import Path
from typing import Iterable, List, Tuple

def iter_python_files(repo_dir: str) -> Iterable[str]:
    """Yield repo-relative paths of .py files (skip venv, build, hidden dirs)."""
    root = Path(repo_dir).resolve()
    skip = {".git", ".venv", "venv", "__pycache__", "build", "dist", ".mypy_cache", ".pytest_cache"}
    for p in root.rglob("*.py"):
        rel = p.relative_to(root).as_posix()
        parts = set(p.parts)
        if any(s in parts for s in skip):
            continue
        if rel.startswith("."):
            continue
        yield rel

def stable_id(*parts: str) -> str:
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]
    return h

def read_lines(repo_dir: str, rel_path: str) -> List[str]:
    p = Path(repo_dir) / rel_path
    try:
        return p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

def snippet_from_lines(lines: List[str], start_line: int, end_line: int, with_lineno: bool=True) -> str:
    start = max(1, start_line)
    end = min(len(lines), max(start, end_line))
    out=[]
    for i in range(start, end+1):
        s = lines[i-1]
        out.append(f"{i:4d}: {s}" if with_lineno else s)
    return "\n".join(out)
