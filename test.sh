export NORM_TYPE='reboot_pre'
python3 test_ppl.py \
    --model_dir /reboot/model_50000 \
    --eval_dataset_path /lpai/volumes/ad-vla-vol-ga/lipengxiang/vla/hf_cache/c4/en \
    --batch_size 1 \
    --max_length 1024 \
    --max_eval_samples 500 \
    --device cuda \
    --dtype bfloat16