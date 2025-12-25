# Repo Data Factory (rule + LLM)

This project generates training data from a local Python repository in two steps:

1. **Rules (deterministic)** produce `*.draft.jsonl` with grounded evidence snippets.
2. **LLM enrichment** rewrites drafts into high-quality training samples while staying strictly grounded.

All prompts and comments are in English.

## Environment

```bash
export RDF_LLM_BASE_URL="http://YOUR_OPENAI_COMPATIBLE_GATEWAY/v1"
export RDF_LLM_API_KEY="sk-..."
export RDF_LLM_MODEL="qwen2.5-instruct"
export RDF_LLM_TIMEOUT_S="60"
# Important if you have HTTP_PROXY pointing to 127.0.0.1:49091 etc.
export RDF_LLM_DISABLE_ENV_PROXY="1"
```

## Scenario 1

Generate drafts:

```bash
python -m repo_data_factory.pipelines.scenario1_rules \
  --repo_dir /path/to/S2AND \
  --out_jsonl ./scenario1.draft.jsonl
```

Enrich with LLM (streaming + resumable, 5-way concurrency):

```bash
python -m repo_data_factory.pipelines.scenario1_llm \
  --in_jsonl ./scenario1.draft.jsonl \
  --out_jsonl ./scenario1.enriched.jsonl \
  --err_log ./scenario1.enriched.errors.log \
  --workers 5
```

## Scenario 2

Generate drafts:

```bash
python -m repo_data_factory.pipelines.scenario2_rules \
  --repo_dir /path/to/S2AND \
  --out_jsonl ./scenario2.draft.jsonl
```

Enrich:

```bash
python -m repo_data_factory.pipelines.scenario2_llm \
  --in_jsonl ./scenario2.draft.jsonl \
  --out_jsonl ./scenario2.enriched.jsonl \
  --err_log ./scenario2.enriched.errors.log \
  --workers 5
```

## Notes on proxy timeouts (127.0.0.1:49091)

If your environment sets `HTTP_PROXY` / `HTTPS_PROXY` to a local proxy, `requests` may route your LLM traffic through it.
This code disables proxy env usage by default (`RDF_LLM_DISABLE_ENV_PROXY=1`) via `session.trust_env = False`.
