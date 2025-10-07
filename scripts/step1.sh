

torchrun \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29501 \
  src/generate_scores.py \
    --input_path data/alpaca_gpt4_data_en.json \
    --output_path data/Data5.0/alpaca_gpt4_data_en-step1.json \
    --quality_model hkust-nlp/deita-quality-scorer \
    --complexity_model hkust-nlp/deita-complexity-scorer \
    --max_len 2048 \
    --batch_size 1



