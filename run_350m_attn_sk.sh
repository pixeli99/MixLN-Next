# Define the set of learning rates and normalization types
norm_type=attn_skip
learning_rates=1e-3
export NORM_TYPE=$norm_type
export POST_NUM=6
export linespace=0.2
export layer_scale='true'
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/lpai/volumes/ad-vla-vol-ga/lipengxiang/vla/hf_cache"

# Function to run a single training task

echo "Training with learning rate: $learning_rates, norm type: $norm_type on GPU $gpu"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=29502 torchrun_main.py \
    --model_config configs/llama_350m.json \
    --lr $learning_rates \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adam \
    --grad_clipping 0.0 \
    --run_name "350m_res_${norm_type}_lr${learning_rates}" \
    --save_dir "${LPAI_MODEL_DIR}/350m_res_${norm_type}_lr${learning_rates}"