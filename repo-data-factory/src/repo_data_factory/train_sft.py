# train_qwen25_sft_repo_data_factory_qlora.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_evidence_block(item: Dict[str, Any], max_chars: int = 8000) -> str:
    ev = item.get("evidence", []) or []
    parts = []
    for i, e in enumerate(ev):
        span = (e or {}).get("span", {}) or {}
        fp = span.get("file_path", "UNKNOWN_FILE")
        sl = span.get("start_line", "?")
        el = span.get("end_line", "?")
        snip = (e or {}).get("snippet", "")
        parts.append(f"[evidence {i}] file={fp} lines={sl}-{el}\n{snip}")
    text = "\n\n".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...<truncated>..."
    return text


def make_prompt(item: Dict[str, Any]) -> Dict[str, str]:
    q = (item.get("question") or "").strip()
    ev = build_evidence_block(item)

    system = (
        "You are a repository-grounded assistant. "
        "Use ONLY the provided evidence snippets. "
        "Do NOT invent APIs, file names, functions, config keys, or behaviors not present in evidence. "
        "If evidence is insufficient, say 'Insufficient evidence' and list what is missing."
    )
    user = f"{q}\n\n--- Evidence ---\n{ev}\n"
    return {"system": system, "user": user}


def make_target(item: Dict[str, Any], include_trace: bool) -> str:
    ans = (item.get("answer") or "").strip()
    if not include_trace:
        return ans

    trace = item.get("trace", [])
    try:
        trace_json = json.dumps(trace, ensure_ascii=False)
    except Exception:
        trace_json = "[]"
    return ans + "\n\n--- RATIONALE_TRACE_JSON ---\n" + trace_json


def build_dataset(paths: List[str], include_trace: bool) -> Dataset:
    all_rows: List[Dict[str, Any]] = []
    for p in paths:
        all_rows.extend(read_jsonl(p))

    clean = []
    for r in all_rows:
        if not isinstance(r, dict):
            continue
        if not (r.get("question") and r.get("answer") and r.get("evidence")):
            continue
        clean.append(r)

    ds = Dataset.from_list(clean)

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        p = make_prompt(ex)
        t = make_target(ex, include_trace=include_trace)
        return {"system": p["system"], "user": p["user"], "target": t}

    return ds.map(_map, remove_columns=ds.column_names)


def to_chat_text(tokenizer: AutoTokenizer, system: str, user: str, target: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train_jsonl", type=str, nargs="+", required=True)
    ap.add_argument("--output_dir", type=str, default="./qwen25-7b-rdf-qlora")
    ap.add_argument("--include_trace", action="store_true")

    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # dtype：优先 bf16（A100/H100/部分 40 系支持），否则 fp16
    if args.bf16:
        compute_dtype = torch.bfloat16
    elif args.fp16:
        compute_dtype = torch.float16
    else:
        # 不指定就让你按环境选；但 QLoRA 计算 dtype 最好还是 bf16/fp16
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = build_dataset(args.train_jsonl, include_trace=args.include_trace)

    def _to_text(ex: Dict[str, Any]) -> Dict[str, Any]:
        return {"text": to_chat_text(tokenizer, ex["system"], ex["user"], ex["target"])}

    ds = ds.map(_to_text, remove_columns=ds.column_names)

    # QLoRA 4bit 配置（NF4 + double quant）
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=compute_dtype,  # 让非 4bit 部分也用这个 dtype
    )

    # 训练前准备：k-bit training +（可选）gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    model.config.use_cache = False  # 否则有些情况下会炸显存/影响 checkpointing

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        optim="paged_adamw_8bit",  # QLoRA 常用
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=train_args,
        packing=True,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[OK] saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
