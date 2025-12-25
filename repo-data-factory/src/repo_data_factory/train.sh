python train_qwen25_sft_repo_data_factory_qlora.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --train_jsonl ./scenario1.enriched.jsonl ./scenario2.enriched.jsonl \
  --output_dir ./qwen25-7b-rdf-qlora \
  --max_seq_len 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --bf16
