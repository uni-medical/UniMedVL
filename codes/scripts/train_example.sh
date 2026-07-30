#!/bin/bash
# Copyright 2025 UniMedVL Team
# SPDX-License-Identifier: Apache-2.0
#
# Example: fine-tune UniMedVL from the released checkpoint on your own data.
#
# Prerequisites:
#   1. Download the checkpoint:
#      huggingface-cli download General-Medical-AI/UniMedVL --local-dir ./checkpoints/UniMedVL
#   2. Fill in your dataset paths in data/dataset_info.py and configure the
#      dataset mixture in data/configs/example.yaml.
#   3. Run from the codes/ directory: bash scripts/train_example.sh

set -e
cd "$(dirname "$0")/.."

# ---- User configuration (override via environment variables) ----
MODEL_PATH=${MODEL_PATH:-./checkpoints/UniMedVL}   # dir with ema.safetensors, ae.safetensors, llm_config.json, vit_config.json, tokenizer files
NUM_GPUS=${NUM_GPUS:-8}                            # with fewer than 8x80GB GPUs, also set --cpu_offload True below to fit optimizer states
DATASET_CONFIG=${DATASET_CONFIG:-data/configs/example.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-output/finetune_example}
MASTER_PORT=${MASTER_PORT:-29503}

torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_addr=127.0.0.1 \
  --master_port=$MASTER_PORT \
  train/pretrain_unimedvl.py \
  --dataset_config_file $DATASET_CONFIG \
  --model_path $MODEL_PATH \
  --resume_from $MODEL_PATH \
  --finetune_from_hf True \
  --finetune_from_ema True \
  --resume_model_only False \
  --auto_resume True \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --visual_gen True \
  --visual_und True \
  --results_dir $OUTPUT_DIR \
  --checkpoint_dir $OUTPUT_DIR/checkpoints \
  --max_checkpoints 2 \
  --total_steps 200 \
  --save_every 200 \
  --log_every 1 \
  --lr 1e-5 \
  --num_workers 1 \
  --expected_num_tokens 18000 \
  --max_num_tokens 20000 \
  --max_num_tokens_per_sample 17000 \
  --text_cond_dropout_prob 0.3 \
  --vit_cond_dropout_prob 0.05 \
  --vae_cond_dropout_prob 0.05 \
  --ce_weight 0.25 \
  --mse_weight 1.0 \
  --ema 0.995 \
  --freeze_llm False \
  --freeze_vit True \
  --freeze_vae True \
  --freeze_und False \
  --copy_init_moe True \
  --num_replicate 1 \
  --num_shard $NUM_GPUS \
  --sharding_strategy HYBRID_SHARD \
  --backward_prefetch BACKWARD_PRE \
  --cpu_offload False
