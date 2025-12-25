from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Set

from tqdm import tqdm

from ..types import QAItem, TraceStep
from ..llm.client import OpenAIChatClient, LLMConfig, extract_first_json_object

SYSTEM_PROMPT = """You are a 'repository training data editor'.
Your task: rewrite a draft QAItem into a high-quality training sample WITHOUT changing facts.

Hard constraints:
1) Use ONLY information found in evidence.snippet. If evidence is insufficient, explicitly say 'insufficient information' in the answer and explain what's missing.
2) Do NOT hallucinate any details: file names, function names, parameter names, default values, return values, exception types, etc.
3) The trace MUST include evidence_refs (indexes into evidence list). Every step must be supported by the evidence.

Output MUST be a JSON object with fields:
- question: string (more natural and specific)
- answer: string (structured, ready for training)
- trace: [{step:int, kind:"extract"|"reason"|"answer", content:string, evidence_refs:[int]}]
""".strip()

def _build_user_prompt(item: QAItem) -> str:
    ev_text=[]
    for i, e in enumerate(item.evidence):
        ev_text.append(
            f"[evidence {i}] file={e.span.file_path} lines={e.span.start_line}-{e.span.end_line}\n{e.snippet}"
        )
    ev_block = "\n\n".join(ev_text)

    return f"""Below is a DRAFT training sample extracted by deterministic rules.
Rewrite it into a high-quality sample under the constraints.

Requirements:
- question: sounds like a real developer question (you may add context but MUST stay within evidence)
- answer: clear conclusion + key conditions / I/O / errors (ONLY if present in evidence)
- trace: 3-6 steps, each step must cite evidence_refs and show reasoning from code to conclusion

--- Draft metadata ---
id: {item.id}
rule_id: {item.rule_id}
title: {item.title}

--- Draft question ---
{item.question}

--- Draft answer ---
{item.answer}

--- Evidence ---
{ev_block}
""".strip()

def _validate_and_apply(item: QAItem, j: Dict) -> QAItem:
    if not isinstance(j, dict) or "question" not in j or "answer" not in j or "trace" not in j:
        raise ValueError("Bad JSON schema (expect question/answer/trace)")

    item.question = str(j["question"]).strip()
    item.answer = str(j["answer"]).strip()

    new_trace=[]
    for t in j["trace"]:
        new_trace.append(TraceStep(
            step=int(t["step"]),
            kind=str(t["kind"]),
            content=str(t["content"]).strip(),
            evidence_refs=[int(x) for x in t.get("evidence_refs", [])],
        ))
    for t in new_trace:
        for r in t.evidence_refs:
            if r < 0 or r >= len(item.evidence):
                raise ValueError(f"Invalid evidence_refs: {r}")
    item.trace = new_trace
    item.meta["llm_enriched"] = True
    return item

def _enrich_one(client: OpenAIChatClient, item: QAItem) -> Tuple[QAItem, Optional[str]]:
    try:
        user = _build_user_prompt(item)
        raw = client.chat_text(SYSTEM_PROMPT, user)
        j = extract_first_json_object(raw)
        item = _validate_and_apply(item, j)
        return item, None
    except Exception as e:
        item.meta["llm_enriched"] = False
        item.meta["llm_error"] = str(e)
        return item, f"{item.id}: {e}"

def _read_jsonl(path: str) -> List[QAItem]:
    items=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            items.append(QAItem.from_dict(json.loads(line)))
    return items

def _read_done_ids(out_jsonl: str) -> Set[str]:
    done=set()
    if not os.path.exists(out_jsonl):
        return done
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d=json.loads(line)
                if "id" in d:
                    done.add(str(d["id"]))
            except Exception:
                continue
    return done

def enrich_items_streaming(
    client: OpenAIChatClient,
    items: List[QAItem],
    out_jsonl: str,
    err_log: str,
    workers: int = 5,
) -> None:
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(err_log) or ".", exist_ok=True)

    done_ids = _read_done_ids(out_jsonl)
    total = len(items)
    remaining = [it for it in items if it.id not in done_ids]
    skipped = total - len(remaining)

    ok=0
    fail=0

    out_f = open(out_jsonl, "a", encoding="utf-8")
    err_f = open(err_log, "a", encoding="utf-8")

    try:
        pbar = tqdm(total=len(remaining), desc="Scenario1 LLM enrich", unit="item")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut2id = {ex.submit(_enrich_one, client, it): it.id for it in remaining}
            for fut in as_completed(fut2id):
                it, err = fut.result()
                out_f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
                out_f.flush()
                if err:
                    err_f.write(err + "\n")
                    err_f.flush()
                    fail += 1
                else:
                    ok += 1
                pbar.update(1)
        pbar.close()
    finally:
        out_f.close()
        err_f.close()

    print(f"[PIPE] total={total} skipped={skipped} ok={ok} fail={fail} out={out_jsonl} err={err_log}", flush=True)

def main() -> int:
    ap = argparse.ArgumentParser(description="Scenario 1: LLM enrichment (streaming, resumable)")
    ap.add_argument("--in_jsonl", required=True, help="Input draft jsonl")
    ap.add_argument("--out_jsonl", required=True, help="Output enriched jsonl (append/resume)")
    ap.add_argument("--err_log", required=True, help="Error log path (append)")
    ap.add_argument("--workers", type=int, default=5, help="Concurrency (threads)")
    args = ap.parse_args()

    cfg = LLMConfig()
    client = OpenAIChatClient(cfg)
    print(f"[DEBUG] LLM base_url={cfg.base_url} model={cfg.model} timeout_s={cfg.timeout_s} disable_env_proxy={cfg.disable_env_proxy}", flush=True)

    items = _read_jsonl(args.in_jsonl)
    print(f"[PIPE] loaded={len(items)} in={args.in_jsonl}", flush=True)

    enrich_items_streaming(client, items, args.out_jsonl, args.err_log, workers=args.workers)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
