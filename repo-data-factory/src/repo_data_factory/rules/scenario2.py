# scenario2.py
# A more stable Scenario2 rule pack with:
# - stricter anchor detection (especially public_api)
# - span-based evidence trimming
# - strong de-duplication & per-(file,kind) frequency control
# - pluggable LLM enrichment to set llm_enriched=True when available
#
# Drop-in concept: rules produce JSON-serializable dict items.

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Repo adapters (YOU ADAPT THIS)
# -----------------------------

@dataclass(frozen=True)
class RepoFile:
    path: str
    lines: List[str]  # 1-based semantics in evidence spans (we store as list[str] 0-based)


@dataclass
class RepoIndex:
    """
    Adapter expected by rules.

    You should implement:
      - iter_files(): Iterable[RepoFile]
      - get_file(path): RepoFile
    """
    files: List[RepoFile]

    def iter_files(self) -> Iterable[RepoFile]:
        return self.files

    def get_file(self, path: str) -> RepoFile:
        for f in self.files:
            if f.path == path:
                return f
        raise KeyError(path)


# -----------------------------
# Output schema helpers
# -----------------------------

@dataclass(frozen=True)
class EvidenceSpan:
    file_path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class EvidenceItem:
    span: EvidenceSpan
    snippet: str


def _mk_snippet(lines: List[str], start_line: int, end_line: int) -> str:
    # start/end are 1-based inclusive
    start_i = max(1, start_line) - 1
    end_i = min(len(lines), end_line)
    buf = []
    for i in range(start_i, end_i):
        ln = i + 1
        buf.append(f"{ln:>4}: {lines[i].rstrip()}")
    return "\n".join(buf)


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# -----------------------------
# Evidence trimming (big win)
# -----------------------------

def trim_span_around_matches(
    repo_file: RepoFile,
    match_lines_1based: List[int],
    *,
    hard_max_lines: int = 120,
    soft_context: int = 35,
) -> Tuple[int, int]:
    """
    Trim huge files/spans to something LLM-friendly:
    - take a window around the densest cluster of matches
    - include a bit of context
    """
    n = len(repo_file.lines)
    if not match_lines_1based:
        # fallback: head
        return 1, min(n, hard_max_lines)

    # cluster matches: choose window center = median
    m = sorted(match_lines_1based)
    center = m[len(m) // 2]
    start = max(1, center - soft_context)
    end = min(n, center + soft_context)

    # ensure within hard max
    while (end - start + 1) > hard_max_lines:
        # shrink symmetrically
        if end - center > center - start:
            end -= 1
        else:
            start += 1

    return start, end


def trim_docstring_plus_key_region(
    repo_file: RepoFile,
    *,
    func_name: str,
    hard_max_lines: int = 140,
) -> Optional[Tuple[int, int]]:
    """
    If we can find `def func_name(`, include:
    - its docstring block (if present)
    - plus a limited region after it
    """
    pat = re.compile(rf"^\s*def\s+{re.escape(func_name)}\s*\(")
    lines = repo_file.lines
    for i, line in enumerate(lines):
        if pat.search(line):
            # 1-based
            def_line = i + 1
            # take a window starting at def_line
            start = def_line
            end = min(len(lines), def_line + hard_max_lines - 1)
            return start, end
    return None


# -----------------------------
# Rule base
# -----------------------------

@dataclass(frozen=True)
class Anchor:
    kind: str
    file_path: str
    anchor_lines: List[int]  # 1-based


@dataclass
class Rule:
    rule_id: str
    title: str

    def find_anchors(self, idx: RepoIndex) -> List[Anchor]:
        raise NotImplementedError

    def build_item(self, idx: RepoIndex, anchor: Anchor) -> Dict[str, Any]:
        raise NotImplementedError


# -----------------------------
# Stable anchor detectors
# -----------------------------

_PUBLIC_API_REEXPORT = re.compile(
    r"^\s*(from\s+\S+\s+import\s+.+|import\s+\S+)\s*(#.*)?$"
)
_PUBLIC_API_ALL = re.compile(r"^\s*__all__\s*=\s*\[")
_PUBLIC_API_COMMENT = re.compile(r"(public\s+api|re-export|reexport|stable api)", re.I)

def detect_public_api_anchors(f: RepoFile) -> Optional[Anchor]:
    """
    Stricter than “__init__.py => public_api”.
    Require at least one of:
      - __all__ assignment
      - explicit re-export pattern in __init__.py (from x import y)
      - comment hinting stability/public API
    """
    if not f.path.endswith("__init__.py"):
        return None

    hit_lines: List[int] = []
    for i, line in enumerate(f.lines):
        if _PUBLIC_API_ALL.search(line):
            hit_lines.append(i + 1)
        elif _PUBLIC_API_REEXPORT.search(line) and "logging" not in line:
            # avoid the common false positive where __init__.py only configures logging
            # (your current output has exactly this problem)
            hit_lines.append(i + 1)
        elif _PUBLIC_API_COMMENT.search(line):
            hit_lines.append(i + 1)

    if not hit_lines:
        return None

    return Anchor(kind="public_api", file_path=f.path, anchor_lines=hit_lines)


_EXT_HINT = re.compile(r"(registry|register|factory|strategy|plugin|extension)", re.I)
def detect_extension_anchors(f: RepoFile) -> List[Anchor]:
    hit_lines = []
    for i, line in enumerate(f.lines):
        if _EXT_HINT.search(line):
            hit_lines.append(i + 1)
    if not hit_lines:
        return []
    return [Anchor(kind="extension_point", file_path=f.path, anchor_lines=hit_lines)]


_FLOW_HINT = re.compile(r"(pipeline|stage|step|first|then|finally|second stage)", re.I)
def detect_flow_anchors(f: RepoFile) -> List[Anchor]:
    hit_lines = []
    for i, line in enumerate(f.lines):
        if _FLOW_HINT.search(line):
            hit_lines.append(i + 1)
    if not hit_lines:
        return []
    return [Anchor(kind="flow_pipeline", file_path=f.path, anchor_lines=hit_lines)]


_PERF_HINT = re.compile(r"(batch|cache|chunk|memory\s+footprint|tqdm|float16|imap|lazy)", re.I)
def detect_perf_anchors(f: RepoFile) -> List[Anchor]:
    hit_lines = []
    for i, line in enumerate(f.lines):
        if _PERF_HINT.search(line):
            hit_lines.append(i + 1)
    if not hit_lines:
        return []
    return [Anchor(kind="performance_scalability", file_path=f.path, anchor_lines=hit_lines)]


_POLICY_HINT = re.compile(r"(special case|fallback|backward-compat|compatibility|TODO\(|policy|rule\s*\()", re.I)
def detect_policy_anchors(f: RepoFile) -> List[Anchor]:
    hit_lines = []
    for i, line in enumerate(f.lines):
        if _POLICY_HINT.search(line):
            hit_lines.append(i + 1)
    if not hit_lines:
        return []
    return [Anchor(kind="algorithm_policy", file_path=f.path, anchor_lines=hit_lines)]


# -----------------------------
# LLM enrichment (pluggable)
# -----------------------------

class LLMClient:
    """
    Very small, pluggable interface.

    Provide either:
      - SCENARIO2_LLM_ENDPOINT: an HTTP endpoint that takes {"prompt": "..."} and returns {"text": "..."}
    or implement your own client and call enrich_items(...) with it.
    """
    def __init__(self, endpoint: str, timeout_s: int = 60):
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def complete(self, prompt: str) -> str:
        # No extra deps: use urllib
        import urllib.request

        payload = json.dumps({"prompt": prompt}).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data.get("text")
        if not isinstance(text, str):
            raise ValueError("LLM endpoint must return JSON with a string field `text`")
        return text


def get_default_llm_client() -> Optional[LLMClient]:
    ep = os.environ.get("SCENARIO2_LLM_ENDPOINT", "").strip()
    if not ep:
        return None
    return LLMClient(ep)


def build_llm_prompt(question: str, evidence: List[EvidenceItem]) -> str:
    ev = "\n\n".join([e.snippet for e in evidence])
    return textwrap.dedent(f"""\
    You are given architecture/design evidence extracted from a local repository.

    Task:
    Answer the question STRICTLY using ONLY the evidence. If evidence is insufficient, say "Insufficient evidence" and list what is missing.

    Question:
    {question}

    Evidence:
    {ev}
    """)


def maybe_llm_enrich_item(item: Dict[str, Any], llm: Optional[LLMClient]) -> Dict[str, Any]:
    """
    If llm is available, generate answer and mark llm_enriched True.
    Otherwise keep the draft answer and llm_enriched False.
    """
    meta = item.get("meta") or {}
    extra = meta.get("extra") or {}
    if llm is None:
        extra["llm_enriched"] = False
        meta["extra"] = extra
        item["meta"] = meta
        return item

    # Build evidence list from item
    evidence_objs: List[EvidenceItem] = []
    for ev in item.get("evidence", []):
        sp = ev.get("span", {})
        snippet = ev.get("snippet", "")
        evidence_objs.append(
            EvidenceItem(
                span=EvidenceSpan(
                    file_path=sp.get("file_path", ""),
                    start_line=int(sp.get("start_line", 1)),
                    end_line=int(sp.get("end_line", 1)),
                ),
                snippet=snippet,
            )
        )

    prompt = build_llm_prompt(item.get("question", ""), evidence_objs)
    try:
        ans = llm.complete(prompt)
        item["answer"] = ans
        extra["llm_enriched"] = True
    except Exception as e:
        # Fail closed: don't break pipeline
        item["answer"] = item.get("answer", "") + f"\n\n[LLM_ENRICH_FAILED] {type(e).__name__}: {e}"
        extra["llm_enriched"] = False

    meta["extra"] = extra
    item["meta"] = meta
    return item


# -----------------------------
# De-dup and frequency control
# -----------------------------

@dataclass
class Scenario2Config:
    max_items_total: int = 200
    max_items_per_file_kind: int = 2  # big lever to stop spam like data.py perf repeated 6x
    evidence_hard_max_lines: int = 140
    supporting_max: int = 2


class Deduper:
    def __init__(self, cfg: Scenario2Config):
        self.cfg = cfg
        self.seen_fingerprints: set[str] = set()
        self.count_by_file_kind: Dict[Tuple[str, str], int] = {}

    def fingerprint(self, item: Dict[str, Any]) -> str:
        rule_id = item.get("rule_id", "")
        anchor_file = item.get("question", "")
        # Also incorporate evidence snippets content hash (stable)
        ev_text = "\n\n".join([e.get("snippet", "") for e in item.get("evidence", [])])
        key = f"{rule_id}||{_norm_ws(anchor_file)[:120]}||{_hash_text(ev_text)}"
        return _hash_text(key)

    def accept(self, item: Dict[str, Any]) -> bool:
        fp = self.fingerprint(item)
        if fp in self.seen_fingerprints:
            return False

        # frequency control by (anchor_file, anchor_kind)
        meta = item.get("meta") or {}
        extra = meta.get("extra") or {}
        anchor_file = extra.get("anchor_file") or ""
        anchor_kind = extra.get("anchor_kind") or extra.get("anchor_kind", "")
        k = (anchor_file, anchor_kind)
        c = self.count_by_file_kind.get(k, 0)
        if anchor_file and anchor_kind and c >= self.cfg.max_items_per_file_kind:
            return False

        self.seen_fingerprints.add(fp)
        if anchor_file and anchor_kind:
            self.count_by_file_kind[k] = c + 1
        return True


# -----------------------------
# Concrete rules (stable variants)
# -----------------------------

class S2ArchRationalePublicAPI(Rule):
    def __init__(self):
        super().__init__(
            rule_id="S2_ARCH_RATIONALE_PUBLIC_API",
            title="Architecture rationale: public API",
        )

    def find_anchors(self, idx: RepoIndex) -> List[Anchor]:
        out: List[Anchor] = []
        for f in idx.iter_files():
            a = detect_public_api_anchors(f)
            if a is not None:
                out.append(a)
        return out

    def build_item(self, idx: RepoIndex, anchor: Anchor) -> Dict[str, Any]:
        f = idx.get_file(anchor.file_path)
        start, end = trim_span_around_matches(
            f, anchor.anchor_lines,
            hard_max_lines=140,
            soft_context=40,
        )
        evidence = [
            EvidenceItem(
                span=EvidenceSpan(anchor.file_path, start, end),
                snippet=_mk_snippet(f.lines, start, end),
            )
        ]
        question = (
            "You are given architecture/design evidence extracted from a local repository.\n"
            "Focus area: Public API surface / stability / exports\n"
            f"Anchor kind: public_api\n"
            f"Anchor file: {anchor.file_path}\n\n"
            "Task:\n"
            "1) Explain what public API surface is visible in the evidence (exports, __all__, re-exports).\n"
            "2) Infer what stability/compatibility constraint this API design implies. If evidence is insufficient, say 'Insufficient evidence' and list what is missing.\n"
            "3) Propose a backward-compatible evolution plan: what to add/change, where (by files/modules seen in evidence), migration steps, tests, risks, rollback.\n\n"
            "Constraints:\n"
            "- Use ONLY the provided evidence snippets.\n"
            "- Do NOT invent new public APIs not present in evidence.\n"
            "- Provide a rationale trace where EACH step cites evidence_refs.\n"
        )
        return {
            "id": f"s2_{_hash_text(anchor.file_path + '|' + anchor.kind)}",
            "scenario": "scenario2",
            "rule_id": self.rule_id,
            "title": self.title,
            "question": question,
            "answer": "(draft) Provide a design rationale and an evolution plan using ONLY the evidence. If evidence is insufficient, say 'Insufficient evidence' and list what is missing.",
            "evidence": [dataclasses.asdict(e) for e in evidence],
            "trace": [
                {"step": 1, "kind": "extract", "content": "Extract what is actually exported/re-exported or declared stable in the evidence.", "evidence_refs": [0]},
                {"step": 2, "kind": "reason", "content": "Infer implied compatibility/stability constraints. If missing, state Insufficient evidence.", "evidence_refs": [0]},
                {"step": 3, "kind": "answer", "content": "Propose backward-compatible evolution (additions only, deprecation path, tests, rollback).", "evidence_refs": [0]},
            ],
            "meta": {"kind": "arch_first_design_task", "extra": {"anchor_kind": anchor.kind, "anchor_file": anchor.file_path, "llm_enriched": False}},
        }


class S2ArchRationaleFlow(Rule):
    def __init__(self):
        super().__init__(rule_id="S2_ARCH_RATIONALE_FLOW", title="Architecture rationale: pipeline/flow")

    def find_anchors(self, idx: RepoIndex) -> List[Anchor]:
        out: List[Anchor] = []
        for f in idx.iter_files():
            out.extend(detect_flow_anchors(f))
        return out

    def build_item(self, idx: RepoIndex, anchor: Anchor) -> Dict[str, Any]:
        f = idx.get_file(anchor.file_path)
        start, end = trim_span_around_matches(
            f, anchor.anchor_lines,
            hard_max_lines=160,
            soft_context=55,
        )
        evidence = [
            EvidenceItem(EvidenceSpan(anchor.file_path, start, end), _mk_snippet(f.lines, start, end))
        ]
        question = (
            "You are given architecture/design evidence extracted from a local repository.\n"
            "Focus area: Flow / pipeline / orchestration\n"
            f"Anchor kind: flow_pipeline\n"
            f"Anchor file: {anchor.file_path}\n\n"
            "Task:\n"
            "1) Explain what pipeline/flow structure is visible in the evidence (stages, boundaries, sequencing).\n"
            "2) Infer the likely goal/constraint this flow is addressing. If evidence is insufficient, say 'Insufficient evidence' and list what is missing.\n"
            "3) Propose an evolution plan: where to integrate a new step safely, compatibility constraints, tests, risks, rollback.\n\n"
            "Constraints:\n"
            "- Use ONLY the provided evidence snippets.\n"
            "- Do NOT invent modules/APIs not present in evidence.\n"
            "- Provide a rationale trace where EACH step cites evidence_refs.\n"
        )
        return {
            "id": f"s2_{_hash_text(anchor.file_path + '|' + anchor.kind)}",
            "scenario": "scenario2",
            "rule_id": self.rule_id,
            "title": self.title,
            "question": question,
            "answer": "(draft) Provide a design rationale and an evolution plan using ONLY the evidence. If evidence is insufficient, say 'Insufficient evidence' and list what is missing.",
            "evidence": [dataclasses.asdict(e) for e in evidence],
            "trace": [
                {"step": 1, "kind": "extract", "content": "Extract stages/sequence/boundaries visible in the evidence.", "evidence_refs": [0]},
                {"step": 2, "kind": "reason", "content": "Infer goal/constraint; if unclear, mark Insufficient evidence and list missing info.", "evidence_refs": [0]},
                {"step": 3, "kind": "answer", "content": "Propose a safe evolution plan: insertion point, compatibility, tests, risks, rollback.", "evidence_refs": [0]},
            ],
            "meta": {"kind": "arch_first_design_task", "extra": {"anchor_kind": anchor.kind, "anchor_file": anchor.file_path, "llm_enriched": False}},
        }


class S2ArchRationalePerf(Rule):
    def __init__(self):
        super().__init__(rule_id="S2_ARCH_RATIONALE_PERF", title="Architecture rationale: performance/scalability")

    def find_anchors(self, idx: RepoIndex) -> List[Anchor]:
        out: List[Anchor] = []
        for f in idx.iter_files():
            out.extend(detect_perf_anchors(f))
        return out

    def build_item(self, idx: RepoIndex, anchor: Anchor) -> Dict[str, Any]:
        f = idx.get_file(anchor.file_path)
        start, end = trim_span_around_matches(
            f, anchor.anchor_lines,
            hard_max_lines=150,
            soft_context=50,
        )
        evidence = [
            EvidenceItem(EvidenceSpan(anchor.file_path, start, end), _mk_snippet(f.lines, start, end))
        ]
        question = (
            "You are given architecture/design evidence extracted from a local repository.\n"
            "Focus area: Performance / scalability / memory / caching / batching\n"
            f"Anchor kind: performance_scalability\n"
            f"Anchor file: {anchor.file_path}\n\n"
            "Task:\n"
            "1) Explain what performance/scalability tradeoffs are visible in the evidence (batching, cache, memory footprint, data structures).\n"
            "2) Infer the likely constraint driving these choices (e.g., memory/time). If evidence is insufficient, say 'Insufficient evidence' and list what is missing.\n"
            "3) Propose an evolution plan: how to improve performance safely, where to change, compatibility constraints, tests, risks, rollback.\n\n"
            "Constraints:\n"
            "- Use ONLY the provided evidence snippets.\n"
            "- Do NOT invent new configuration keys or flags not present in evidence.\n"
            "- Provide a rationale trace where EACH step cites evidence_refs.\n"
        )
        return {
            "id": f"s2_{_hash_text(anchor.file_path + '|' + anchor.kind)}",
            "scenario": "scenario2",
            "rule_id": self.rule_id,
            "title": self.title,
            "question": question,
            "answer": "(draft) Provide a design rationale and an evolution plan using ONLY the evidence. If evidence is insufficient, say 'Insufficient evidence' and list what is missing.",
            "evidence": [dataclasses.asdict(e) for e in evidence],
            "trace": [
                {"step": 1, "kind": "extract", "content": "Extract concrete perf patterns (batching/cache/chunk/memory notes) from evidence.", "evidence_refs": [0]},
                {"step": 2, "kind": "reason", "content": "Infer the constraint (memory/time/IO). If unclear, Insufficient evidence.", "evidence_refs": [0]},
                {"step": 3, "kind": "answer", "content": "Propose a safe perf evolution plan: where to change, keep compatibility, tests, risks, rollback.", "evidence_refs": [0]},
            ],
            "meta": {"kind": "arch_first_design_task", "extra": {"anchor_kind": anchor.kind, "anchor_file": anchor.file_path, "llm_enriched": False}},
        }


class S2ArchRationalePolicy(Rule):
    def __init__(self):
        super().__init__(rule_id="S2_ARCH_RATIONALE_POLICY", title="Architecture rationale: algorithmic policy")

    def find_anchors(self, idx: RepoIndex) -> List[Anchor]:
        out: List[Anchor] = []
        for f in idx.iter_files():
            out.extend(detect_policy_anchors(f))
        return out

    def build_item(self, idx: RepoIndex, anchor: Anchor) -> Dict[str, Any]:
        f = idx.get_file(anchor.file_path)
        start, end = trim_span_around_matches(
            f, anchor.anchor_lines,
            hard_max_lines=160,
            soft_context=55,
        )
        evidence = [
            EvidenceItem(EvidenceSpan(anchor.file_path, start, end), _mk_snippet(f.lines, start, end))
        ]
        question = (
            "You are given architecture/design evidence extracted from a local repository.\n"
            "Focus area: Algorithmic policy / special cases / fallback / compatibility hacks\n"
            f"Anchor kind: algorithm_policy\n"
            f"Anchor file: {anchor.file_path}\n\n"
            "Task:\n"
            "1) Explain what policy/strategy is visible in the evidence (rules, special cases, fallbacks, merging decisions).\n"
            "2) Infer the likely requirement/constraint this policy addresses. If evidence is insufficient, say 'Insufficient evidence' and list what is missing.\n"
            "3) Propose an evolution plan: how to adjust/replace policies safely, where to change, compatibility, tests, risks, rollback.\n\n"
            "Constraints:\n"
            "- Use ONLY the provided evidence snippets.\n"
            "- Do NOT invent behavior not present in evidence.\n"
            "- Provide a rationale trace where EACH step cites evidence_refs.\n"
        )
        return {
            "id": f"s2_{_hash_text(anchor.file_path + '|' + anchor.kind)}",
            "scenario": "scenario2",
            "rule_id": self.rule_id,
            "title": self.title,
            "question": question,
            "answer": "(draft) Provide a design rationale and an evolution plan using ONLY the evidence. If evidence is insufficient, say 'Insufficient evidence' and list what is missing.",
            "evidence": [dataclasses.asdict(e) for e in evidence],
            "trace": [
                {"step": 1, "kind": "extract", "content": "Extract explicit policies/special cases/fallbacks visible in evidence.", "evidence_refs": [0]},
                {"step": 2, "kind": "reason", "content": "Infer which constraint the policy targets; if unclear, Insufficient evidence.", "evidence_refs": [0]},
                {"step": 3, "kind": "answer", "content": "Propose safe evolution: how to replace/adjust, tests, compatibility, risks, rollback.", "evidence_refs": [0]},
            ],
            "meta": {"kind": "arch_first_design_task", "extra": {"anchor_kind": anchor.kind, "anchor_file": anchor.file_path, "llm_enriched": False}},
        }


# Optional: extension rule (kept generic, but stable)
class S2ArchRationaleExtension(Rule):
    def __init__(self):
        super().__init__(rule_id="S2_ARCH_RATIONALE_EXTENSION", title="Architecture rationale: extension mechanism")

    def find_anchors(self, idx: RepoIndex) -> List[Anchor]:
        out: List[Anchor] = []
        for f in idx.iter_files():
            out.extend(detect_extension_anchors(f))
        return out

    def build_item(self, idx: RepoIndex, anchor: Anchor) -> Dict[str, Any]:
        f = idx.get_file(anchor.file_path)
        start, end = trim_span_around_matches(
            f, anchor.anchor.anchor_lines if False else anchor.anchor_lines,  # keep simple if refactor
            hard_max_lines=160,
            soft_context=55,
        )
        evidence = [
            EvidenceItem(EvidenceSpan(anchor.file_path, start, end), _mk_snippet(f.lines, start, end))
        ]
        question = (
            "You are given architecture/design evidence extracted from a local repository.\n"
            "Focus area: Extension mechanism / registry / factory / strategy\n"
            f"Anchor kind: extension_point\n"
            f"Anchor file: {anchor.file_path}\n\n"
            "Task:\n"
            "1) Explain what architecture/design choice is visible in the evidence.\n"
            "2) Infer the most likely goal/requirement/constraint this design is addressing (reverse-infer). If evidence is insufficient, say 'Insufficient evidence' and list what is missing.\n"
            "3) Propose an evolution plan aligned with the architecture: what to change/add, where (by files/modules seen in evidence), how to keep compatibility, what to test, risks, and rollback.\n\n"
            "Constraints:\n"
            "- Use ONLY the provided evidence snippets.\n"
            "- Do NOT invent APIs/mechanisms not present in evidence.\n"
            "- Provide a rationale trace where EACH step cites evidence_refs.\n"
        )
        return {
            "id": f"s2_{_hash_text(anchor.file_path + '|' + anchor.kind)}",
            "scenario": "scenario2",
            "rule_id": self.rule_id,
            "title": self.title,
            "question": question,
            "answer": "(draft) Provide a design rationale and evolution plan using ONLY the evidence. If evidence is insufficient, say 'Insufficient evidence' and list what is missing.",
            "evidence": [dataclasses.asdict(e) for e in evidence],
            "trace": [
                {"step": 1, "kind": "extract", "content": "Extract extension/registry/factory/strategy signals from evidence.", "evidence_refs": [0]},
                {"step": 2, "kind": "reason", "content": "Reverse-infer likely goal/constraint; if unclear, Insufficient evidence.", "evidence_refs": [0]},
                {"step": 3, "kind": "answer", "content": "Propose aligned evolution plan: add new impl via existing mechanism, compat, tests, risks, rollback.", "evidence_refs": [0]},
            ],
            "meta": {"kind": "arch_first_design_task", "extra": {"anchor_kind": anchor.kind, "anchor_file": anchor.file_path, "llm_enriched": False}},
        }


# -----------------------------
# Entry point
# -----------------------------

def get_scenario2_rules() -> List[Rule]:
    # Order matters: public_api first (rare, strict), then extension/flow/perf/policy
    return [
        S2ArchRationalePublicAPI(),
        S2ArchRationaleExtension(),
        S2ArchRationaleFlow(),
        S2ArchRationalePerf(),
        S2ArchRationalePolicy(),
    ]


def run_scenario2_rules(
    idx: RepoIndex,
    *,
    cfg: Optional[Scenario2Config] = None,
    llm: Optional[LLMClient] = None,
) -> List[Dict[str, Any]]:
    """
    Main function you call from pipeline.
    - Generates items
    - Applies dedup + frequency control
    - Optionally does LLM enrichment
    """
    cfg = cfg or Scenario2Config()
    llm = llm if llm is not None else get_default_llm_client()

    rules = get_scenario2_rules()
    deduper = Deduper(cfg)

    items: List[Dict[str, Any]] = []
    for rule in rules:
        for anchor in rule.find_anchors(idx):
            item = rule.build_item(idx, anchor)
            # attach anchor info consistently for dedupe + control
            meta = item.get("meta") or {}
            extra = meta.get("extra") or {}
            extra["anchor_kind"] = anchor.kind
            extra["anchor_file"] = anchor.file_path
            meta["extra"] = extra
            item["meta"] = meta

            if not deduper.accept(item):
                continue

            # enrich (optional)
            item = maybe_llm_enrich_item(item, llm)

            items.append(item)
            if len(items) >= cfg.max_items_total:
                return items

    return items


# -----------------------------
# Minimal JSONL writer helper
# -----------------------------

def write_jsonl(path: str, items: Sequence[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return len(items)

# -----------------------------
# Back-compat entrypoint for pipeline
# scenario2_rules.py expects: from ..rules.scenario2 import run_rules
# -----------------------------

def _build_repo_index_from_dir(repo_dir: str) -> RepoIndex:
    """
    Minimal filesystem-based indexer.
    This avoids depending on internal repo scanners, so it will work immediately.
    """
    import os

    files: List[RepoFile] = []
    skip_dirs = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"}
    # You can tune this list. Keep it conservative: focus on code/docs.
    keep_ext = {".py", ".md", ".txt", ".rst"}

    for root, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, repo_dir)

            _, ext = os.path.splitext(fn)
            if ext.lower() not in keep_ext and fn.lower() not in {"readme", "readme.md"}:
                continue

            try:
                # Skip huge files defensively
                if os.path.getsize(path) > 2_000_000:  # 2MB
                    continue
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.read().splitlines()
            except Exception:
                continue

            files.append(RepoFile(path=rel.replace("\\", "/"), lines=lines))

    return RepoIndex(files=files)


def run_rules(
    *,
    repo_dir: str,
    out_jsonl: str,
    # Compatibility: pipeline may pass this (you hit this before)
    max_cli_reqs: Optional[int] = None,  # accepted but not used here
    # Allow future pipeline args without breaking
    **kwargs: Any,
) -> int:
    """
    Pipeline entrypoint.

    Expected by: repo_data_factory.pipelines.scenario2_rules
    Writes JSONL and returns number of rows.
    """
    # Build index
    idx = _build_repo_index_from_dir(repo_dir)

    # Config knobs (optional via kwargs)
    cfg = Scenario2Config(
        max_items_total=int(kwargs.get("max_items_total", 200)),
        max_items_per_file_kind=int(kwargs.get("max_items_per_file_kind", 2)),
        evidence_hard_max_lines=int(kwargs.get("evidence_hard_max_lines", 140)),
        supporting_max=int(kwargs.get("supporting_max", 2)),
    )

    # Optional LLM endpoint (env var already supported; allow override via kwargs too)
    llm = None
    llm_endpoint = kwargs.get("llm_endpoint")
    if isinstance(llm_endpoint, str) and llm_endpoint.strip():
        llm = LLMClient(llm_endpoint.strip())
    else:
        llm = get_default_llm_client()

    items = run_scenario2_rules(idx, cfg=cfg, llm=llm)
    n = write_jsonl(out_jsonl, items)
    return n