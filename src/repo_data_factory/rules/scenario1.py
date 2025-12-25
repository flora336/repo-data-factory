from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from ..types import QAItem, TraceStep
from ..utils import stable_id
from .common import make_evidence, iter_py_files, read_lines

# =============================================================================
# Scenario 1: "Business process / rules" Q&A grounded in local repo code.
#
# Rules are deterministic extractors:
# - They find "questionable" code points (configs, validations, errors, tests...)
# - They attach minimal evidence spans (file + line range + snippet)
# - They emit draft QAItems (question/answer/trace are placeholders)
#
# LLM stage (separate pipeline) then rewrites these drafts into high-quality
# training samples without changing facts.
# =============================================================================


# -----------------------------
# Shared templates (English)
# -----------------------------

_DRAFT_ANSWER = "(draft) Summarize the behavior/rule implied by the evidence snippet(s)."

_TRACE_TEMPLATE = [
    TraceStep(step=1, kind="extract", content="Locate the relevant code evidence.", evidence_refs=[0]),
    TraceStep(step=2, kind="reason", content="Derive the rule/behavior from conditions, branches, and calls in evidence.", evidence_refs=[0]),
    TraceStep(step=3, kind="answer", content="State the conclusion and key triggers/inputs/outputs based on evidence.", evidence_refs=[0]),
]


def _mk_item(
    rule_id: str,
    title: str,
    question: str,
    evidence: List,
    meta: dict,
) -> QAItem:
    item_id = f"s1_{stable_id(rule_id, title, evidence[0].span.file_path, str(evidence[0].span.start_line))}"
    return QAItem(
        id=item_id,
        scenario="scenario1",
        rule_id=rule_id,
        title=title,
        question=question,
        answer=_DRAFT_ANSWER,
        evidence=evidence,
        trace=list(_TRACE_TEMPLATE),
        meta=meta,
    )


# =============================================================================
# Rule S1_R1: Environment / getenv-based config
# =============================================================================

_ENV_GET_RE = re.compile(r"os\.getenv\(\s*[\"']([A-Z0-9_]+)[\"']\s*,?")
_ENV_ENVIRON_GET_RE = re.compile(r"os\.environ\.get\(\s*[\"']([A-Z0-9_]+)[\"']\s*,?")
_ENV_SUBSCRIPT_RE = re.compile(r"os\.environ\[\s*[\"']([A-Z0-9_]+)[\"']\s*\]")


def rule_s1_config_env(repo_dir: Path, max_per_file: int = 30) -> List[QAItem]:
    items: List[QAItem] = []
    for fpath in iter_py_files(repo_dir):
        rel = str(fpath.relative_to(repo_dir))
        lines = read_lines(fpath)
        hit = 0
        for i, line in enumerate(lines, start=1):
            m = _ENV_GET_RE.search(line) or _ENV_ENVIRON_GET_RE.search(line) or _ENV_SUBSCRIPT_RE.search(line)
            if not m:
                continue
            key = m.group(1)
            evidence = [make_evidence(rel, i, i, repo_dir)]
            title = f"Config from env: {key}"
            q = f"In this repository, what behavior/rule is controlled by environment variable `{key}`? Explain using the given code evidence."
            meta = {"kind": "config_env", "extra": {"key": key}}
            items.append(_mk_item("S1_CONFIG_ENV", title, q, evidence, meta))
            hit += 1
            if hit >= max_per_file:
                break
    return items


# =============================================================================
# Rule S1_R2: Public-facing docstring/API contract snippets
# =============================================================================

_DOCSTRING_START_RE = re.compile(r'^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[\(:]')
_TRIPLE_QUOTE_RE = re.compile(r'^\s*(["\']{3})')


def rule_s1_docstring_contract(repo_dir: Path, max_items: int = 300) -> List[QAItem]:
    items: List[QAItem] = []
    for fpath in iter_py_files(repo_dir):
        rel = str(fpath.relative_to(repo_dir))
        lines = read_lines(fpath)
        for idx in range(len(lines) - 1):
            m = _DOCSTRING_START_RE.match(lines[idx])
            if not m:
                continue
            name = m.group(2)
            # Look for immediate docstring block (very lightweight heuristic)
            if idx + 1 >= len(lines):
                continue
            if not _TRIPLE_QUOTE_RE.match(lines[idx + 1]):
                continue
            # Find docstring end within next ~30 lines
            end = None
            quote = _TRIPLE_QUOTE_RE.match(lines[idx + 1]).group(1)
            for j in range(idx + 2, min(len(lines), idx + 32)):
                if quote in lines[j]:
                    end = j + 1
                    break
            if end is None:
                continue
            start_line = idx + 1
            end_line = end
            evidence = [make_evidence(rel, start_line, end_line, repo_dir)]
            title = f"Docstring contract: {name}"
            q = f"What does `{name}` claim to do according to its docstring/contract? Summarize the contract using only the evidence."
            meta = {"kind": "docstring_contract", "extra": {"name": name}}
            items.append(_mk_item("S1_DOCSTRING_CONTRACT", title, q, evidence, meta))
            if len(items) >= max_items:
                return items
    return items


# =============================================================================
# Rule S1_R3: Exceptions / raise statements
# =============================================================================

_RAISE_RE = re.compile(r"^\s*raise\s+([A-Za-z_][A-Za-z0-9_]*)")


def rule_s1_raise_exceptions(repo_dir: Path, max_per_file: int = 30) -> List[QAItem]:
    items: List[QAItem] = []
    for fpath in iter_py_files(repo_dir):
        rel = str(fpath.relative_to(repo_dir))
        lines = read_lines(fpath)
        hit = 0
        for i, line in enumerate(lines, start=1):
            m = _RAISE_RE.match(line)
            if not m:
                continue
            exc = m.group(1)
            evidence = [make_evidence(rel, i, min(i + 2, len(lines)), repo_dir)]
            title = f"Exception behavior: raise {exc}"
            q = f"Under what condition does the code raise `{exc}` here, and what does that imply about expected inputs/state?"
            meta = {"kind": "raise_exception", "extra": {"exception": exc}}
            items.append(_mk_item("S1_RAISE_EXCEPTION", title, q, evidence, meta))
            hit += 1
            if hit >= max_per_file:
                break
    return items


# =============================================================================
# Rule S1_R4: Input validation patterns (assert / if ...: raise ...)
# =============================================================================

_ASSERT_RE = re.compile(r"^\s*assert\s+(.+?)(?:,|$)")
_IF_RAISE_RE = re.compile(r"^\s*if\s+(.+?)\s*:\s*$")


def rule_s1_validation(repo_dir: Path, max_per_file: int = 30) -> List[QAItem]:
    items: List[QAItem] = []
    for fpath in iter_py_files(repo_dir):
        rel = str(fpath.relative_to(repo_dir))
        lines = read_lines(fpath)
        hit = 0
        for i, line in enumerate(lines, start=1):
            m = _ASSERT_RE.match(line)
            if m:
                cond = m.group(1).strip()
                evidence = [make_evidence(rel, i, i, repo_dir)]
                title = "Validation via assert"
                q = f"What validation does this `assert` enforce (`{cond}`), and what would happen if it fails?"
                meta = {"kind": "validation_assert", "extra": {"condition": cond}}
                items.append(_mk_item("S1_VALIDATION", title, q, evidence, meta))
                hit += 1
            else:
                m2 = _IF_RAISE_RE.match(line)
                if m2 and i < len(lines):
                    # if next non-empty line starts with raise
                    j = i + 1
                    while j <= len(lines) and lines[j - 1].strip() == "":
                        j += 1
                    if j <= len(lines) and lines[j - 1].lstrip().startswith("raise"):
                        cond = m2.group(1).strip()
                        evidence = [make_evidence(rel, i, min(j, i + 2), repo_dir)]
                        title = "Validation via if/raise"
                        q = f"What invalid condition is being checked (`{cond}`), and what is the expected behavior when it is met?"
                        meta = {"kind": "validation_if_raise", "extra": {"condition": cond}}
                        items.append(_mk_item("S1_VALIDATION", title, q, evidence, meta))
                        hit += 1
            if hit >= max_per_file:
                break
    return items


# =============================================================================
# Rule S1_R5: Logging rules (logger.* / logging.*)
# =============================================================================

_LOG_RE = re.compile(r"\b(?:logger|logging)\.(debug|info|warning|error|exception)\(")


def rule_s1_logging(repo_dir: Path, max_per_file: int = 30) -> List[QAItem]:
    items: List[QAItem] = []
    for fpath in iter_py_files(repo_dir):
        rel = str(fpath.relative_to(repo_dir))
        lines = read_lines(fpath)
        hit = 0
        for i, line in enumerate(lines, start=1):
            m = _LOG_RE.search(line)
            if not m:
                continue
            level = m.group(1)
            evidence = [make_evidence(rel, i, i, repo_dir)]
            title = f"Logging at {level}"
            q = f"What event/state is logged at `{level}` level here, and what does that tell us about the business flow?"
            meta = {"kind": "logging", "extra": {"level": level}}
            items.append(_mk_item("S1_LOGGING", title, q, evidence, meta))
            hit += 1
            if hit >= max_per_file:
                break
    return items


# =============================================================================
# Rule S1_R6: CLI / __main__ entrypoints
# =============================================================================

_MAIN_RE = re.compile(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:')


def rule_s1_entrypoint_main(repo_dir: Path, max_items: int = 100) -> List[QAItem]:
    items: List[QAItem] = []
    for fpath in iter_py_files(repo_dir):
        rel = str(fpath.relative_to(repo_dir))
        lines = read_lines(fpath)
        for i, line in enumerate(lines, start=1):
            if _MAIN_RE.search(line):
                evidence = [make_evidence(rel, i, min(i + 8, len(lines)), repo_dir)]
                title = f"Entrypoint: {rel}"
                q = f"What is the program entrypoint behavior in `{rel}`? Describe what gets executed when run as a script."
                meta = {"kind": "entrypoint_main", "extra": {"file": rel}}
                items.append(_mk_item("S1_ENTRYPOINT_MAIN", title, q, evidence, meta))
                if len(items) >= max_items:
                    return items
                break
    return items


# =============================================================================
# Rule S1_R7: Tests (pytest-style)
# =============================================================================

_PYTEST_RE = re.compile(r"^\s*def\s+(test_[A-Za-z0-9_]+)\s*\(")


def rule_s1_pytest(repo_dir: Path, max_items: int = 300) -> List[QAItem]:
    items: List[QAItem] = []
    for fpath in iter_py_files(repo_dir):
        rel = str(fpath.relative_to(repo_dir))
        if "test" not in rel.lower():
            continue
        lines = read_lines(fpath)
        for i, line in enumerate(lines, start=1):
            m = _PYTEST_RE.match(line)
            if not m:
                continue
            name = m.group(1)
            evidence = [make_evidence(rel, i, min(i + 12, len(lines)), repo_dir)]
            title = f"Test behavior: {name}"
            q = f"What behavior or rule does test `{name}` verify? Summarize the expected behavior using only the evidence."
            meta = {"kind": "pytest", "extra": {"test_name": name}}
            items.append(_mk_item("S1_PYTEST", title, q, evidence, meta))
            if len(items) >= max_items:
                return items
    return items


# =============================================================================
# Rule S1_R8: File I/O patterns (open/read/write)
# =============================================================================

_OPEN_RE = re.compile(r"\bopen\(")
_PATHLIB_IO_RE = re.compile(r"\bPath\([^)]+\)\.(read_text|write_text|open)\(")


def rule_s1_file_io(repo_dir: Path, max_per_file: int = 20) -> List[QAItem]:
    items: List[QAItem] = []
    for fpath in iter_py_files(repo_dir):
        rel = str(fpath.relative_to(repo_dir))
        lines = read_lines(fpath)
        hit = 0
        for i, line in enumerate(lines, start=1):
            if _OPEN_RE.search(line) or _PATHLIB_IO_RE.search(line):
                evidence = [make_evidence(rel, i, min(i + 2, len(lines)), repo_dir)]
                title = "File I/O behavior"
                q = "What file I/O behavior is performed here (read/write path, format, side effects) based on the evidence?"
                meta = {"kind": "file_io"}
                items.append(_mk_item("S1_FILE_IO", title, q, evidence, meta))
                hit += 1
                if hit >= max_per_file:
                    break
    return items


# =============================================================================
# Rule S1_R9: Regex patterns (re.compile / re.match / re.search)
# =============================================================================

_REGEX_RE = re.compile(r"\bre\.(compile|match|search|findall)\(")


def rule_s1_regex(repo_dir: Path, max_per_file: int = 20) -> List[QAItem]:
    items: List[QAItem] = []
    for fpath in iter_py_files(repo_dir):
        rel = str(fpath.relative_to(repo_dir))
        lines = read_lines(fpath)
        hit = 0
        for i, line in enumerate(lines, start=1):
            if _REGEX_RE.search(line):
                evidence = [make_evidence(rel, i, i, repo_dir)]
                title = "Regex rule"
                q = "What pattern matching rule is applied here, and what inputs would it accept/reject (based on evidence only)?"
                meta = {"kind": "regex"}
                items.append(_mk_item("S1_REGEX", title, q, evidence, meta))
                hit += 1
                if hit >= max_per_file:
                    break
    return items


# =============================================================================
# Rule S1_R10: Async / concurrency patterns (async def / await)
# =============================================================================

_ASYNC_DEF_RE = re.compile(r"^\s*async\s+def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_AWAIT_RE = re.compile(r"\bawait\b")


def rule_s1_async(repo_dir: Path, max_items: int = 150) -> List[QAItem]:
    items: List[QAItem] = []
    for fpath in iter_py_files(repo_dir):
        rel = str(fpath.relative_to(repo_dir))
        lines = read_lines(fpath)
        for i, line in enumerate(lines, start=1):
            m = _ASYNC_DEF_RE.match(line)
            if not m:
                continue
            name = m.group(1)
            # grab a few lines including await if nearby
            end = min(i + 12, len(lines))
            evidence = [make_evidence(rel, i, end, repo_dir)]
            title = f"Async behavior: {name}"
            q = f"What asynchronous behavior does `{name}` implement (awaited calls, sequencing) based on the evidence?"
            meta = {"kind": "async", "extra": {"name": name}}
            items.append(_mk_item("S1_ASYNC", title, q, evidence, meta))
            if len(items) >= max_items:
                return items
    return items


# =============================================================================
# Orchestrator
# =============================================================================

_RULE_FUNCS = [
    rule_s1_config_env,
    rule_s1_docstring_contract,
    rule_s1_raise_exceptions,
    rule_s1_validation,
    rule_s1_logging,
    rule_s1_entrypoint_main,
    rule_s1_pytest,
    rule_s1_file_io,
    rule_s1_regex,
    rule_s1_async,
]


def run_rules(repo_dir: Path) -> List[QAItem]:
    """Run all Scenario 1 rules and return draft QAItems."""
    items: List[QAItem] = []
    for fn in _RULE_FUNCS:
        try:
            items.extend(fn(repo_dir))
        except Exception as e:
            # Keep deterministic stage robust: one broken rule shouldn't kill the pipeline.
            # Errors are surfaced via metadata for later inspection.
            items.append(
                QAItem(
                    id=f"s1_{stable_id('RULE_ERROR', fn.__name__)}",
                    scenario="scenario1",
                    rule_id="S1_RULE_ERROR",
                    title=f"Rule failure: {fn.__name__}",
                    question=f"Rule `{fn.__name__}` failed to run.",
                    answer=str(e),
                    evidence=[],
                    trace=[],
                    meta={"kind": "rule_error", "extra": {"rule": fn.__name__, "error": str(e)}},
                )
            )
    return items
