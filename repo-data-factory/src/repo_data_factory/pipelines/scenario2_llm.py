# src/repo_data_factory/pipelines/scenario2_llm.py
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ..llm.client import LLMConfig, OpenAIChatClient
from ..types import QAItem, TraceStep

# =============================================================================
# Scenario 2 LLM enrichment:
#
# Modes:
# - requirement-driven: question is already provided. MUST NOT change question.
# - intent-driven (meta.extra.llm_generate_question=True): LLM generates question + answer + trace
#
# Grounding rules:
# - ONLY use evidence snippets
# - If insufficient evidence, must say "Insufficient evidence" and list missing info
# - trace steps must cite evidence_refs (indices into evidence list)
#
# This version hardens:
# - robust JSON extraction
# - auto-fix trace length (4~8) + step renumber
# - evidence budget truncation to reduce prompt bloat
# - thread-local OpenAIChatClient for concurrency safety
# - richer error logging
# - optional question rollback instead of failing whole item
# =============================================================================


# -----------------------------
# Tunables (safe defaults)
# -----------------------------

# Prompt size control (character-based; simple, effective)
MAX_EVIDENCE_CHARS_TOTAL = 12000   # total evidence characters across all snippets
MAX_EVIDENCE_CHARS_PER_SNIPPET = 2500

# LLM retry (for transient bad JSON / minor formatting issues)
MAX_LLM_ATTEMPTS = 3
RETRY_BACKOFF_S = 0.8

# Trace constraints
MIN_TRACE_STEPS = 4
MAX_TRACE_STEPS = 8

# If False, question change in requirement-driven will fail the item.
# If True, we keep old question but still accept answer/trace.
ALLOW_QUESTION_ROLLBACK = True

# Store head of raw LLM response on failure to debug
RAW_HEAD_CHARS_FOR_DEBUG = 800


SYSTEM_PROMPT = """
You are a "repository training data editor" specialized in ARCHITECTURE-AWARE DESIGN + DESIGN-INTENT tasks.

You will receive a draft QAItem with evidence snippets.

Two modes:
A) requirement-driven: question is already provided. You MUST NOT change question.
B) intent-driven: question is a draft placeholder. You MUST generate a precise, engineer-like question
   about the PURPOSE / RATIONALE of the design evidenced.

Hard constraints:
1) You may ONLY use information present in evidence snippets.
   If evidence is insufficient, you MUST say exactly: "Insufficient evidence" and explicitly list what is missing.
2) You MUST NOT invent file names, module names, function names, argument names, default values,
   return values, exception types, config keys, or behaviors not literally present in evidence.
3) You MUST output a rationale trace with 4~8 steps. Each step MUST have evidence_refs (indices into evidence list).
4) If there are >=2 evidence snippets, your trace MUST cite at least 2 different evidence indices overall.
5) Output MUST be a single JSON object, with no markdown, no extra text.

Output JSON schema (exact fields):
{
  "question": "string",
  "answer": "string",
  "trace": [
    {"step": 1, "kind": "extract"|"reason"|"answer", "content": "string", "evidence_refs": [0,1]}
  ],
  "grounding_warnings": ["string", ...]   // optional
}

Answer style requirements:
- Provide sections: Overview, Evidence-backed intent (why), Proposed Changes (if any), Compatibility, Risks, Tests, Rollback.
- If the task is mainly explanation (intent), Proposed Changes can be "No change proposed" unless evidence supports changes.
""".strip()


# -----------------------------
# Thread-local client
# -----------------------------
_thread_local = threading.local()


def _get_client(cfg: LLMConfig) -> OpenAIChatClient:
    c = getattr(_thread_local, "client", None)
    if c is None:
        _thread_local.client = OpenAIChatClient(cfg)
    return _thread_local.client


# -----------------------------
# Prompt building with budget
# -----------------------------
def _truncate_text(s: str, limit: int) -> str:
    s = s or ""
    if len(s) <= limit:
        return s
    # keep head+tail a bit helps
    head = int(limit * 0.75)
    tail = limit - head
    return s[:head] + "\n...<truncated>...\n" + s[-tail:]


def _build_user_prompt(item: QAItem) -> str:
    extra = item.meta.get("extra", {}) if isinstance(item.meta, dict) else {}
    llm_generate_question = bool(extra.get("llm_generate_question", False))
    mode_line = (
        "MODE: intent-driven (LLM must generate question)"
        if llm_generate_question
        else "MODE: requirement-driven (LLM must keep question unchanged)"
    )

    intent_hint = ""
    if llm_generate_question:
        hint = extra.get("intent_hint", "")
        if isinstance(hint, str) and hint.strip():
            intent_hint = (
                "\n\n--- Intent hint (for phrasing only; MUST still be grounded in evidence) ---\n"
                + hint.strip()
                + "\n"
            )

    # Evidence budget: trim per-snippet then cap total
    ev_blocks: List[str] = []
    total_chars = 0

    for i, e in enumerate(item.evidence):
        snippet = _truncate_text(str(e.snippet), MAX_EVIDENCE_CHARS_PER_SNIPPET)
        block = (
            f"[evidence {i}] file={e.span.file_path} lines={e.span.start_line}-{e.span.end_line}\n"
            f"{snippet}"
        )
        if total_chars + len(block) > MAX_EVIDENCE_CHARS_TOTAL:
            # add a final note and stop
            ev_blocks.append(
                f"[evidence {i}] <omitted: evidence budget exceeded; add narrower spans if needed>"
            )
            break
        ev_blocks.append(block)
        total_chars += len(block)

    ev_block = "\n\n".join(ev_blocks)

    return f"""
Rewrite the following draft training sample into a high-quality "architecture-aware design" sample.

{mode_line}

Rules:
- Use ONLY the evidence below.
- Output a single JSON object (no markdown, no extra commentary).
- Trace must have 4~8 steps; every step must cite evidence_refs.
- If MODE is requirement-driven, DO NOT change the question text (only trivial whitespace cleanup).

--- Draft metadata ---
id: {item.id}
rule_id: {item.rule_id}
title: {item.title}

--- Draft question ---
{item.question}

--- Draft answer ---
{item.answer}
{intent_hint}
--- Evidence ---
{ev_block}
""".strip()


# -----------------------------
# JSON extraction (robust)
# -----------------------------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("LLM returned empty response")

    # 1) direct JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) fenced JSON
    m = _JSON_FENCE_RE.search(raw)
    if m:
        cand = m.group(1).strip()
        obj = json.loads(cand)
        if isinstance(obj, dict):
            return obj

    # 3) substring from first { to last }
    s = raw.find("{")
    e = raw.rfind("}")
    if s != -1 and e != -1 and e > s:
        cand = raw[s : e + 1]
        obj = json.loads(cand)
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"LLM did not return valid JSON. Head={raw[:200]!r}")


def _normalize_q(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


# -----------------------------
# Trace auto-fix and validation
# -----------------------------
def _renumber_trace_steps(tr: List[Dict[str, Any]]) -> None:
    for i, t in enumerate(tr, start=1):
        t["step"] = i


def _pad_trace_to_min_steps(tr: List[Dict[str, Any]], evidence_len: int) -> List[Dict[str, Any]]:
    """
    Ensure trace length >= MIN_TRACE_STEPS by inserting a "reason" step.
    """
    if len(tr) >= MIN_TRACE_STEPS:
        return tr

    # pick a safe evidence ref
    default_ref = [0] if evidence_len > 0 else [0]
    for t in tr:
        refs = t.get("evidence_refs")
        if isinstance(refs, list) and refs:
            try:
                default_ref = [int(refs[0])]
                break
            except Exception:
                pass

    # Insert before last step if possible (keep final as answer-ish)
    insert_at = max(0, len(tr) - 1)
    tr.insert(
        insert_at,
        {
            "step": 0,
            "kind": "reason",
            "content": (
                "If the evidence does not explicitly show an extension/registry/factory/strategy mechanism, "
                "state 'Insufficient evidence' and list what is missing (e.g., registration/selection logic, "
                "interfaces/contracts, or where implementations are wired)."
            ),
            "evidence_refs": default_ref,
        },
    )
    return tr


def _cap_trace_to_max_steps(tr: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(tr) <= MAX_TRACE_STEPS:
        return tr
    # Keep first (extract), middle reasons, and last (answer). Simple trim.
    keep_head = 2
    keep_tail = 1
    middle = tr[keep_head : len(tr) - keep_tail]
    middle = middle[: max(0, MAX_TRACE_STEPS - keep_head - keep_tail)]
    return tr[:keep_head] + middle + tr[-keep_tail:]


def _validate_trace_and_build(item: QAItem, tr_any: Any) -> List[TraceStep]:
    if not isinstance(tr_any, list) or not tr_any:
        raise ValueError("Bad JSON schema: trace must be a non-empty list")

    # normalize to list[dict]
    tr: List[Dict[str, Any]] = []
    for t in tr_any:
        if isinstance(t, dict):
            tr.append(t)
        else:
            raise ValueError("Bad trace element: not an object")

    tr = _pad_trace_to_min_steps(tr, evidence_len=len(item.evidence))
    tr = _cap_trace_to_max_steps(tr)
    _renumber_trace_steps(tr)

    new_trace: List[TraceStep] = []
    for t in tr:
        step = int(t.get("step", 0))
        kind = t.get("kind", "")
        content = str(t.get("content", "")).strip()
        refs = t.get("evidence_refs", [])

        if kind not in ("extract", "reason", "answer"):
            raise ValueError(f"Bad trace kind: {kind}")
        if not isinstance(refs, list) or not refs:
            raise ValueError("Each trace step must have non-empty evidence_refs")

        refs_i = []
        for x in refs:
            refs_i.append(int(x))

        for r in refs_i:
            if r < 0 or r >= len(item.evidence):
                raise ValueError(f"Invalid evidence_refs: {r} out of range")

        if not content:
            raise ValueError("Trace step content must be non-empty")

        new_trace.append(TraceStep(step=step, kind=kind, content=content, evidence_refs=refs_i))

    # Enforce multi-evidence grounding when possible
    all_refs = set()
    for ts in new_trace:
        all_refs.update(ts.evidence_refs)

    if len(item.evidence) >= 2 and len(all_refs) < 2:
        raise ValueError("Trace must cite at least 2 different evidence snippets when evidence>=2")

    item.meta["llm_trace_unique_evidence_count"] = len(all_refs)
    return new_trace


def _apply_llm_obj(item: QAItem, obj: Dict[str, Any]) -> QAItem:
    if not isinstance(obj, dict):
        raise ValueError("Bad JSON: not an object")
    if "question" not in obj or "answer" not in obj or "trace" not in obj:
        raise ValueError("Bad JSON schema: missing question/answer/trace")

    # remove confusing flag in extra
    extra = item.meta.get("extra", {}) if isinstance(item.meta, dict) else {}
    if isinstance(extra, dict) and "llm_enriched" in extra:
        extra.pop("llm_enriched", None)

    allow_q = bool(extra.get("llm_generate_question", False))

    new_q = str(obj["question"]).strip()
    old_q = (item.question or "").strip()

    if not allow_q:
        if _normalize_q(old_q) != _normalize_q(new_q):
            if ALLOW_QUESTION_ROLLBACK:
                # keep old question, still accept answer/trace
                item.meta["llm_question_rolled_back"] = True
            else:
                raise ValueError("Question changed but llm_generate_question is False")

    item.question = new_q if allow_q else old_q
    item.answer = str(obj["answer"]).strip()

    item.trace = _validate_trace_and_build(item, obj["trace"])

    if "grounding_warnings" in obj and isinstance(obj["grounding_warnings"], list):
        item.meta["grounding_warnings"] = [str(x) for x in obj["grounding_warnings"]]

    item.meta["llm_enriched"] = True
    return item


# -----------------------------
# LLM call with retries
# -----------------------------
def _call_llm_with_retries(cfg: LLMConfig, item: QAItem) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Returns (obj_or_none, raw_text).
    """
    client = _get_client(cfg)
    user = _build_user_prompt(item)

    last_raw = ""
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        raw = client.chat_text(SYSTEM_PROMPT, user)
        last_raw = raw or ""
        try:
            obj = _extract_json_object(last_raw)
            return obj, last_raw
        except Exception:
            # retry with small backoff
            if attempt < MAX_LLM_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_S * attempt)
                continue
            return None, last_raw

    return None, last_raw


def _enrich_one(cfg: LLMConfig, item: QAItem) -> Tuple[QAItem, Optional[str]]:
    """
    Return (updated_item, err_or_none)
    """
    # count attempts for debugging
    item.meta["llm_attempts"] = 0
    item.meta["llm_enriched"] = False

    # remove confusing extra.llm_enriched early
    extra = item.meta.get("extra", {}) if isinstance(item.meta, dict) else {}
    if isinstance(extra, dict) and "llm_enriched" in extra:
        extra.pop("llm_enriched", None)

    try:
        for _ in range(MAX_LLM_ATTEMPTS):
            item.meta["llm_attempts"] += 1
            obj, raw = _call_llm_with_retries(cfg, item)
            if obj is None:
                # cannot parse JSON
                head = (raw or "").strip()[:RAW_HEAD_CHARS_FOR_DEBUG]
                return item, f"{item.id}: LLM returned non-JSON or unparsable JSON. raw_head={head!r}"

            try:
                item2 = _apply_llm_obj(item, obj)
                return item2, None
            except Exception as ve:
                # validation failed; add more context and stop (usually deterministic)
                head = json.dumps(obj, ensure_ascii=False)[:RAW_HEAD_CHARS_FOR_DEBUG]
                return item, f"{item.id}: ValidationError: {ve}. obj_head={head!r}"

        return item, f"{item.id}: Unknown error: exceeded attempts"
    except Exception as e:
        return item, f"{item.id}: Exception: {e}"


# -----------------------------
# IO helpers
# -----------------------------
def _read_items_jsonl(path: str) -> List[QAItem]:
    items: List[QAItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(QAItem.from_dict(data))
    return items


def _load_done_ids(out_jsonl: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(out_jsonl):
        return done
    try:
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "id" in obj:
                        done.add(str(obj["id"]))
                except Exception:
                    continue
    except Exception:
        return done
    return done


def enrich_items_streaming_concurrent(
    cfg: LLMConfig,
    items: List[QAItem],
    out_jsonl: str,
    err_log: str,
    workers: int = 5,
) -> Tuple[int, int, int]:
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(err_log) or ".", exist_ok=True)

    done_ids = _load_done_ids(out_jsonl)
    remaining = [it for it in items if it.id not in done_ids]

    total = len(items)
    ok = 0
    fail = 0

    with open(out_jsonl, "a", encoding="utf-8") as outf, open(err_log, "a", encoding="utf-8") as errf:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_enrich_one, cfg, it): it for it in remaining}
            pbar = tqdm(total=len(remaining), desc=f"Scenario2 LLM enrich (workers={workers})")

            for fut in as_completed(futures):
                it0 = futures[fut]
                try:
                    updated, err = fut.result()
                except Exception as e:
                    updated = it0
                    updated.meta["llm_enriched"] = False
                    err = f"{it0.id}: FutureException: {e}"

                outf.write(json.dumps(asdict(updated), ensure_ascii=False) + "\n")
                outf.flush()

                if err:
                    # include rule_id + anchor file if available
                    extra = updated.meta.get("extra", {}) if isinstance(updated.meta, dict) else {}
                    anchor = ""
                    if isinstance(extra, dict):
                        anchor = f" anchor_file={extra.get('anchor_file','')}"
                    errf.write(f"{err}{anchor}\n")
                    errf.flush()
                    fail += 1
                else:
                    ok += 1

                pbar.update(1)

            pbar.close()

    return total, ok, fail


# -----------------------------
# CLI
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", type=str, required=True, help="Scenario2 draft jsonl from rules.")
    parser.add_argument("--out_jsonl", type=str, required=True, help="Output enriched jsonl (append/resumable).")
    parser.add_argument("--err_log", type=str, required=True, help="Error log txt (append).")
    parser.add_argument("--workers", type=int, default=5, help="Concurrency level (default=5).")
    args = parser.parse_args()

    items = _read_items_jsonl(args.in_jsonl)

    cfg = LLMConfig()
    print(f"[PIPE] total={len(items)} out={args.out_jsonl} err={args.err_log} workers={args.workers}", flush=True)
    print(f"[LLM] model={cfg.model}", flush=True)
    print(f"[LLM] base_url={cfg.base_url}", flush=True)
    print(f"[LLM] timeout_s={cfg.timeout_s}", flush=True)
    print(
        f"[CFG] MAX_EVIDENCE_CHARS_TOTAL={MAX_EVIDENCE_CHARS_TOTAL} "
        f"MAX_EVIDENCE_CHARS_PER_SNIPPET={MAX_EVIDENCE_CHARS_PER_SNIPPET} "
        f"attempts={MAX_LLM_ATTEMPTS} "
        f"trace_steps={MIN_TRACE_STEPS}~{MAX_TRACE_STEPS} "
        f"ALLOW_QUESTION_ROLLBACK={ALLOW_QUESTION_ROLLBACK}",
        flush=True,
    )

    total, ok, fail = enrich_items_streaming_concurrent(
        cfg=cfg,
        items=items,
        out_jsonl=args.out_jsonl,
        err_log=args.err_log,
        workers=args.workers,
    )

    print(f"[DONE] total={total} ok={ok} fail={fail} out={args.out_jsonl} err={args.err_log}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())