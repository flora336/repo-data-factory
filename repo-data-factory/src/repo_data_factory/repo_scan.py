# src/repo_data_factory/repo_scan.py
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .utils import iter_python_files, read_lines, snippet_from_lines

# ---------------------------
# Data structures
# ---------------------------

@dataclass(frozen=True)
class SymbolDef:
    """A top-level symbol definition discovered from AST."""
    name: str
    kind: str  # "function" | "class" | "assign"
    file_path: str  # repo-relative path
    start_line: int
    end_line: int

@dataclass
class ParsedPyFile:
    """Parsed representation of a Python file (repo-relative)."""
    file_path: str                 # repo-relative path, e.g. "s2and/consts.py"
    module: str                    # module name, e.g. "s2and.consts"
    source: str                    # full file source text
    lines: List[str]               # source splitlines()
    tree: Optional[ast.AST]        # AST or None if parse failed

    # Imports (raw module strings)
    imports: List[str] = field(default_factory=list)                 # e.g. ["os", "json", "s2and.data"]
    from_imports: List[Tuple[str, List[str]]] = field(default_factory=list)  # e.g. [("s2and.data", ["ANDData"])]

    # Top-level symbol defs
    symbols: List[SymbolDef] = field(default_factory=list)

@dataclass
class RepoIndex:
    """
    Repository-wide index used by rules (especially Scenario2).

    Why this exists:
    - Scenario1 rules are mostly local-to-snippet and can work with simple file iteration.
    - Scenario2 rules often need cross-file anchors: import hubs, entrypoints, API surfaces, flows.
    """
    repo_dir: str
    files: Dict[str, ParsedPyFile]                     # key: repo-relative file_path
    module_to_file: Dict[str, str]                     # key: module, value: file_path
    import_edges: List[Tuple[str, str]]                # (src_module, dst_module)
    symbol_index: Dict[str, List[SymbolDef]]           # name -> list[SymbolDef]

    def get_file(self, rel_path: str) -> Optional[ParsedPyFile]:
        return self.files.get(rel_path)

    def get_by_module(self, module: str) -> Optional[ParsedPyFile]:
        fp = self.module_to_file.get(module)
        return self.files.get(fp) if fp else None

    def import_out_degree(self) -> Dict[str, int]:
        """Out-degree by module: how many distinct repo modules a module imports."""
        deg: Dict[str, Set[str]] = {}
        for src, dst in self.import_edges:
            deg.setdefault(src, set()).add(dst)
        return {k: len(v) for k, v in deg.items()}

    def import_in_degree(self) -> Dict[str, int]:
        """In-degree by module: how many distinct repo modules import this module."""
        deg: Dict[str, Set[str]] = {}
        for src, dst in self.import_edges:
            deg.setdefault(dst, set()).add(src)
        return {k: len(v) for k, v in deg.items()}

# ---------------------------
# Helpers
# ---------------------------

def _to_module_name(rel_path: str) -> str:
    """
    Convert repo-relative path to a python-ish module name.
    Examples:
      - "s2and/consts.py"       -> "s2and.consts"
      - "s2and/__init__.py"     -> "s2and"
      - "a/b/__init__.py"       -> "a.b"
      - "scripts/run_x.py"      -> "scripts.run_x"
    """
    p = Path(rel_path)
    parts = list(p.parts)
    if not parts:
        return ""
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]  # drop .py
    # collapse __init__ to package name
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([x for x in parts if x])

def _parse_imports(tree: ast.AST) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    imports: List[str] = []
    from_imports: List[Tuple[str, List[str]]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # node.module can be None: "from . import x"
            mod = node.module or ""
            names = [a.name for a in node.names if a.name]
            from_imports.append((mod, names))

    return imports, from_imports

def _collect_top_level_symbols(rel_path: str, tree: ast.AST) -> List[SymbolDef]:
    out: List[SymbolDef] = []
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(SymbolDef(
                name=node.name,
                kind="function",
                file_path=rel_path,
                start_line=int(getattr(node, "lineno", 1) or 1),
                end_line=int(getattr(node, "end_lineno", getattr(node, "lineno", 1) or 1) or 1),
            ))
        elif isinstance(node, ast.ClassDef):
            out.append(SymbolDef(
                name=node.name,
                kind="class",
                file_path=rel_path,
                start_line=int(getattr(node, "lineno", 1) or 1),
                end_line=int(getattr(node, "end_lineno", getattr(node, "lineno", 1) or 1) or 1),
            ))
        elif isinstance(node, ast.Assign):
            # only capture simple NAME = ... at top-level
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id:
                    out.append(SymbolDef(
                        name=tgt.id,
                        kind="assign",
                        file_path=rel_path,
                        start_line=int(getattr(node, "lineno", 1) or 1),
                        end_line=int(getattr(node, "end_lineno", getattr(node, "lineno", 1) or 1) or 1),
                    ))
    return out

def _is_repo_module(mod: str, module_to_file: Dict[str, str]) -> bool:
    """
    Treat as "repo module" if it is a package/module we have a file for.
    We also consider submodules: importing "s2and.data" is a repo module if present.
    """
    if not mod:
        return False
    if mod in module_to_file:
        return True
    # also allow prefix match: if "s2and" exists, "s2and.xxx" is likely internal
    top = mod.split(".")[0]
    return top in module_to_file

def _build_import_edges(files: Dict[str, ParsedPyFile], module_to_file: Dict[str, str]) -> List[Tuple[str, str]]:
    edges: Set[Tuple[str, str]] = set()
    for pf in files.values():
        src = pf.module
        if not src:
            continue

        # import x.y
        for mod in pf.imports:
            # keep as-is; later rules can decide if they want only internal
            if _is_repo_module(mod, module_to_file):
                edges.add((src, mod))

        # from x.y import A
        for mod, _names in pf.from_imports:
            if _is_repo_module(mod, module_to_file):
                edges.add((src, mod))

    return sorted(edges)

# ---------------------------
# Public APIs
# ---------------------------

def build_repo_index(repo_dir: str) -> RepoIndex:
    """
    Build a repository index for advanced rules (Scenario2).

    This is intentionally conservative:
    - if a file fails to parse, we still keep the file with tree=None
    - we do not do any heavy static analysis (no type inference, no full call graph)
    """
    root = Path(repo_dir).resolve()
    files: Dict[str, ParsedPyFile] = {}
    module_to_file: Dict[str, str] = {}
    symbol_index: Dict[str, List[SymbolDef]] = {}

    # First pass: load files, compute module names
    for rel_path in iter_python_files(str(root)):
        module = _to_module_name(rel_path)
        abs_path = root / rel_path
        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            source = ""
        lines = source.splitlines()

        tree: Optional[ast.AST] = None
        imports: List[str] = []
        from_imports: List[Tuple[str, List[str]]] = []
        symbols: List[SymbolDef] = []

        if source.strip():
            try:
                tree = ast.parse(source)
                imports, from_imports = _parse_imports(tree)
                symbols = _collect_top_level_symbols(rel_path, tree)
            except Exception:
                tree = None

        pf = ParsedPyFile(
            file_path=rel_path,
            module=module,
            source=source,
            lines=lines,
            tree=tree,
            imports=imports,
            from_imports=from_imports,
            symbols=symbols,
        )
        files[rel_path] = pf
        if module:
            module_to_file[module] = rel_path

        for s in symbols:
            symbol_index.setdefault(s.name, []).append(s)

    # Second pass: import graph edges (internal-only)
    import_edges = _build_import_edges(files, module_to_file)

    return RepoIndex(
        repo_dir=str(root),
        files=files,
        module_to_file=module_to_file,
        import_edges=import_edges,
        symbol_index=symbol_index,
    )

def get_snippet(repo: RepoIndex, rel_path: str, start_line: int, end_line: int) -> str:
    """Convenience wrapper for producing numbered snippets from RepoIndex."""
    pf = repo.get_file(rel_path)
    if not pf:
        return ""
    return snippet_from_lines(pf.lines, start_line, end_line, with_lineno=True)

# ---------------------------
# Backward-compatible lightweight scan
# ---------------------------

@dataclass
class RepoScanResult:
    """Lightweight scan result (kept for backward compatibility)."""
    repo_dir: str
    py_files: List[str]

def scan_repo(repo_dir: str) -> RepoScanResult:
    """
    Lightweight scan API.

    Kept so existing code that imports scan_repo doesn't break.
    Scenario2 should prefer build_repo_index() for architecture-style rules.
    """
    return RepoScanResult(repo_dir=repo_dir, py_files=list(iter_python_files(repo_dir)))

def iter_repo_files(repo_dir: str) -> Iterable[str]:
    """
    Backward-compatible alias.
    Prefer utils.iter_python_files() directly if you don't need this indirection.
    """
    yield from iter_python_files(repo_dir)

def read_repo_snippet(repo_dir: str, rel_path: str, start_line: int, end_line: int) -> str:
    """Backward-compatible snippet helper."""
    lines = read_lines(repo_dir, rel_path)
    return snippet_from_lines(lines, start_line, end_line, with_lineno=True)