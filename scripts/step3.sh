





torchrun \
  --nproc_per_node=6 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29507 \
  src2/ICL-influence.py \
    --data_path data/wizardv1-filtered_with_input-step2.json \
    --save_path data/wizardv1-filtered_with_input-step3.json \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --max_length 4096 \


