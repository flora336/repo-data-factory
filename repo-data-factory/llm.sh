export RDF_LLM_BASE_URL="http://172.22.121.5:3000/v1"
export RDF_LLM_API_KEY="sk-waevYYOAbwuMj7poC5Ae07E5D17a4140Ae6233B5F64cF937"
#export RDF_LLM_BASE_URL="http://172.22.121.55:9997/v1"
#export RDF_LLM_API_KEY="sk-VVtOjibnkTEjhLKl36Ef465eE21d438bA38976E617688f39"
export RDF_LLM_MODEL="qwen2.5-instruct"
sce_num=2
python -m repo_data_factory.pipelines.scenario${sce_num}_llm \
  --in_jsonl ./scenario${sce_num}.draft.jsonl \
  --out_jsonl ./scenario${sce_num}.enriched.jsonl \
  --err_log ./scenario${sce_num}.enriched.errors.log \
  --workers 5


