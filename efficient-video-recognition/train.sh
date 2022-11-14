#!/usr/bin/env sh
# sudo kill -9 PID

exp_dir=runs/k400_vitb16_16f_dec4x768_exp2

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 3 \
  main.py \
    --num_steps 500 \
    --save_freq 5 \
    --eval_freq 5 \
    --print_freq 5 \
    --backbone "ViT-B/16-lnpre" \
    --backbone_type clip \
    --backbone_path /storage/hieunmt/zaloai_liveness/download/ViT-B-16.pt \
    --pretrain k400_b16_pretrain/pretrain-checkpoint.pth \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 768 \
    --decoder_num_heads 12 \
    --label_smoothing 0.1 \
    --num_classes 2 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --train_list_path ../train.txt \
    --val_list_path ../val.txt \
    --batch_size 54 \
    --batch_split 1 \
    --auto_augment rand-m7-n2-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 0 \
    --num_frames 16 \
    --sampling_rate 8 \
    --num_spatial_views 3 \
    --num_temporal_views 3 \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --disable_fp16 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"