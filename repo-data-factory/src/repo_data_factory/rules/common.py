from __future__ import annotations

import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from ..types import CodeSpan, Evidence


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_lines(path: Path) -> List[str]:
    return read_text(path).splitlines()


def format_snippet(lines: List[str], start_line: int, end_line: int) -> str:
    """Return a snippet with '  {lineno}: ' prefixes (1-based)."""
    start_line = max(1, start_line)
    end_line = max(start_line, end_line)
    out: List[str] = []
    for ln in range(start_line, min(end_line, len(lines)) + 1):
        out.append(f"{ln:4d}: {lines[ln-1]}")
    return "\n".join(out)


def make_evidence(file_path: str, start_line: int, end_line: int, repo_dir: Path) -> Evidence:
    p = repo_dir / file_path
    lines = read_lines(p)
    snippet = format_snippet(lines, start_line, end_line)
    return Evidence(span=CodeSpan(file_path=file_path, start_line=start_line, end_line=end_line), snippet=snippet)


def iter_py_files(repo_dir: Path) -> Iterator[Path]:
    """Yield Python files under repo_dir (excluding typical build/venv dirs)."""
    skip_dirs = {".git", ".venv", "venv", "__pycache__", "build", "dist", ".mypy_cache", ".pytest_cache"}
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for fn in files:
            if fn.endswith(".py"):
                yield Path(root) / fn


_IMPORT_RE = re.compile(r"^\s*(?:from\s+([a-zA-Z0-9_\.]+)\s+import\s+|import\s+([a-zA-Z0-9_\.]+))")


def build_import_graph(repo_dir: Path) -> Dict[str, Dict[str, int]]:
    """Build a very lightweight import graph: module -> {imported_module: count}."""
    graph: Dict[str, Dict[str, int]] = {}
    for fpath in iter_py_files(repo_dir):
        rel = str(fpath.relative_to(repo_dir))
        lines = read_lines(fpath)
        for line in lines:
            m = _IMPORT_RE.match(line)
            if not m:
                continue
            mod = m.group(1) or m.group(2)
            if not mod:
                continue
            graph.setdefault(rel, {})
            graph[rel][mod] = graph[rel].get(mod, 0) + 1
    return graph


def top_import_hubs(import_graph: Dict[str, Dict[str, int]], top_k: int = 10) -> List[Tuple[str, int]]:
    """Return (file, degree) sorted by degree, using out-degree as a proxy."""
    deg: List[Tuple[str, int]] = []
    for file, imports in import_graph.items():
        deg.append((file, sum(imports.values())))
    deg.sort(key=lambda x: x[1], reverse=True)
    return deg[:top_k]


def find_lines_matching(repo_dir: Path, file_rel: str, pattern: re.Pattern) -> List[int]:
    """Return 1-based line numbers matching the regex."""
    p = repo_dir / file_rel
    lines = read_lines(p)
    hits: List[int] = []
    for i, line in enumerate(lines, start=1):
        if pattern.search(line):
            hits.append(i)
    return hits
