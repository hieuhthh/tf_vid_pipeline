#!/usr/bin/env sh
# sudo kill -9 PID

exp_dir=runs/k400_vitb16_16f_dec4x768
weight_path=${exp_dir}/best-checkpoint.pth

mkdir -p "${exp_dir}"
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.run --nproc_per_node 1 --master_port=6969 \
  main.py \
    --infer_only \
    --pretrain $weight_path \
    --backbone "ViT-B/16-lnpre" \
    --backbone_type clip \
    --backbone_path ../download/ViT-B-16.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 768 \
    --decoder_num_heads 12 \
    --num_classes 2 \
    --checkpoint_dir "${exp_dir}" \
    --val_list_path ../test.txt \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 0 \
    --num_frames 16 \
    --sampling_rate 16 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
  2>&1 | tee "${exp_dir}/infer-$(date +"%Y%m%d_%H%M%S").log"
