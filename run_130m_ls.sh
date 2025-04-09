# Define the set of learning rates and normalization types
norm_type=cod
learning_rates=1e-3
export NORM_TYPE=$norm_type
export POST_NUM=6
export linespace=0.2
export layer_scale='true'
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/lpai/volumes/ad-vla-vol-ga/lipengxiang/vla/hf_cache"
# Function to run a single training task

echo "Training with learning rate: $learning_rates, norm type: $norm_type on GPU $gpu"

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 --master_port=29510 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr $learning_rates \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adam \
    --grad_clipping 0.0 \
    --run_name "130m_res_${norm_type}_lr${learning_rates}_layer_scale" \
    --save_dir "130m_res_${norm_type}_lr${learning_rates}"