
python src/probe-retrival.py \
  --input_path data/alpaca_gpt4_data_en-step1.json \
  --output_path data/apaca_gpt4_data_en-step2.json \
  --sbert_model all-mpnet-base-v2 \
  --top5 5 \
  --topN 32 \
  --k 5 \
  --save_embed_path data/alpaca.npy \
  --sbert_batch_size 256 \
  --device cuda \
  --faiss_gpu \
