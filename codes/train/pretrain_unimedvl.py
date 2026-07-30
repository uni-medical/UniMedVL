# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0   

"""
Unified NAVIT Training Script

This script implements unified multimodal training for vision-language models
with FSDP distributed training support.

Key Features:
- Three-stage training: VAE alignment → Discriminator → Joint training
- Unified checkpoint save/load interface
- Support for both visual understanding and generation
"""

import functools
import glob
import shutil
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import contextlib
from typing import Optional
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time as time_second
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn as nn

timestamp = time_second()
local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(timestamp))

import torch
torch.backends.cudnn.benchmark = False
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae, AutoEncoder
from modeling.unimedvl import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, fsdp_ema_setup_with_ddp, fsdp_ema_update, fsdp_wrapper_with_ddp
)




def log_model_parameters(logger, model, language_model, vit_model, vae_model, training_args):
    """Log parameter statistics for each model component."""

    def count_params(model, model_name):
        """Count total, trainable, and frozen parameters."""
        if model is None:
            return 0, 0, 0

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        logger.info(f"{model_name}: Total {total_params/1e6:.1f}M | Trainable {trainable_params/1e6:.1f}M | Frozen {frozen_params/1e6:.1f}M")
        return total_params, trainable_params, frozen_params

    logger.info("=" * 60)
    logger.info("Model Parameter Statistics")
    logger.info("=" * 60)

    llm_total, llm_trainable, _ = count_params(language_model, "Language Model (LLM)")

    vit_total = vit_trainable = 0
    if training_args.visual_und and vit_model is not None:
        if hasattr(model, 'vit_model'):
            vit_total, vit_trainable, _ = count_params(model.vit_model, "Vision Transformer (ViT)")
        else:
            vit_total, vit_trainable, _ = count_params(vit_model, "Vision Transformer (ViT)")

    vae_total = vae_trainable = 0
    if training_args.visual_gen:
        if hasattr(model, 'vae_model') and model.vae_model is not None:
            vae_total, vae_trainable, _ = count_params(model.vae_model, "VAE (Internal)")
        elif vae_model is not None:
            vae_total, vae_trainable, _ = count_params(vae_model, "VAE (External)")

    total_all, trainable_all, frozen_all = count_params(model, "Complete Bagel Model")

    logger.info("-" * 60)
    logger.info(f"Summary: LLM {llm_total/1e6:.1f}M | ViT {vit_total/1e6:.1f}M | VAE {vae_total/1e6:.1f}M | Total {total_all/1e6:.1f}M")
    logger.info(f"Total Trainable Parameters: {trainable_all/1e6:.1f}M ({trainable_all/1e9:.2f}B)")
    logger.info("=" * 60)


@dataclass
class ModelArguments:
    """Model configuration arguments."""
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer_with_hidden_outputs",
        metadata={"help": "Python class name of the load layer to instantiate."}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )


@dataclass
class DataArguments:
    """Data loading and processing arguments."""
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:

    # Modality switches
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )

    # Bookkeeping & logging
    max_checkpoints: int = field(
        default=5,
        metadata={"help": "Maximum number of checkpoints to keep in the checkpoint directory."}
    )
    results_dir: str = field(
        default="output",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_name: str = field(
        default="med_bagel_project",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    enable_tensorboard: bool = field(
        default=True,
        metadata={"help": "Enable TensorBoard logging alongside W&B."}
    )

    # Reproducibility & resume
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=True,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)."}
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    resume_model_optimizer: bool = field(
        default=True,
        metadata={"help": "Load model weights and optimizer states."}
    )

    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HuggingFace model."}
    )

    # Reporting frequency
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # Optimization & scheduler
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )

    # Component-specific learning rates (currently unused, reserved for future extensions)
    e2e_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Reserved parameter for component-specific learning rates. Currently unused."}
    )
    vae_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Independent learning rate for VAE parameters. If None, uses main lr."}
    )
    vit_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Independent learning rate for ViT parameters. If None, uses main lr."}
    )
    llm_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Independent learning rate for language model parameters. If None, uses main lr."}
    )

    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )

    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )

    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )

    max_grad_norm: int = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )

    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )

    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )

    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )

    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )

    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )

    # Distributed training / FSDP
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )

    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )

    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )

    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )

    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # Module freezing
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )

    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )

    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don't fine-tune encoder/decoder."}
    )

    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )

    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )

    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )



def main():
    """Main training function."""
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set default learning rates if not specified
    startup_msgs = []
    if training_args.llm_lr is None:
        training_args.llm_lr = training_args.lr
        if dist.get_rank() == 0:
            startup_msgs.append(f"LLM learning rate not set, using default: {training_args.llm_lr}")
    if training_args.vae_lr is None:
        training_args.vae_lr = training_args.lr
        if dist.get_rank() == 0:
            startup_msgs.append(f"VAE learning rate not set, using default: {training_args.vae_lr}")
    if training_args.vit_lr is None:
        training_args.vit_lr = training_args.lr
        if dist.get_rank() == 0:
            startup_msgs.append(f"ViT learning rate not set, using default: {training_args.vit_lr}")

    # Setup logging
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())

        if training_args.enable_tensorboard:
            tensorboard_dir = os.path.join(training_args.checkpoint_dir, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tensorboard_dir)
        else:
            writer = None
    else:
        logger = create_logger(None, dist.get_rank())
        writer = None
    dist.barrier()
    
    if dist.get_rank() == 0:
        for m in startup_msgs:
            logger.info(m)
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # Prepare auto resume logic
    
    # First, try to find the latest checkpoint for auto-resume
    latest_checkpoint = None
    is_resuming_from_interruption = False
    
    if training_args.auto_resume:
        latest_checkpoint = get_latest_ckpt(training_args.checkpoint_dir)
        if latest_checkpoint is not None:
            is_resuming_from_interruption = True
            logger.info(f"🔄 Auto-resume: Found checkpoint {latest_checkpoint}")
        else:
            logger.info(f"🆕 Auto-resume: No checkpoint found in {training_args.checkpoint_dir}")
    
    # Determine the actual resume configuration based on the scenario
    if is_resuming_from_interruption:
        # SCENARIO 1: Resuming from interruption (checkpoint_dir has content)
        # In this case, we want full training state recovery
        resume_from = latest_checkpoint
        resume_model_only = training_args.resume_model_only   
        finetune_from_ema = False  
        logger.info("📋 Resume mode: CONTINUE TRAINING (from interruption)")
        logger.info("   → Loading full training state (model + optimizer + scheduler)")
        logger.info("   → finetune_from_ema automatically disabled")
        
    else:
        # SCENARIO 2: Starting new training (possibly from pretrained model)
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only

        if resume_from is not None:
            finetune_from_ema = training_args.finetune_from_ema

            if resume_model_only:
                mode = "FINETUNE FROM EMA" if finetune_from_ema else "FINETUNE FROM MODEL"
            else:
                mode = "RESUME FROM EMA" if finetune_from_ema else "RESUME FROM MODEL"
            logger.info(f"📋 Resume mode: {mode}")
            logger.info(f"   → Loading from: {resume_from}")
        else:
            finetune_from_ema = False
            logger.info("📋 Resume mode: FRESH START")

    # Validate checkpoint resume configuration
    if finetune_from_ema and (resume_from is None or not os.path.exists(resume_from)):
        error_msg = (
            "CONFIGURATION ERROR: finetune_from_ema=True requires a valid resume_from path!\n"
            f"   Current resume_from: {resume_from}\n"
            "   \n"
            "   To fix this, you need to:\n"
            "   1. Provide a valid checkpoint path: --resume_from /path/to/checkpoint\n"
            "   2. Ensure the checkpoint folder contains ema.safetensors\n"
            "   \n"
            "   Example correct usage:\n"
            "   --auto_resume False \\\n"
            "   --resume_from /path/to/checkpoint \\\n"
            "   --resume_model_only True \\\n"
            "   --finetune_from_ema True\n"
            "   \n"
            "   Without resume_from, there's no checkpoint folder to load ema.safetensors from!"
        )
        logger.error(error_msg)
        raise ValueError("finetune_from_ema=True requires valid resume_from path")

    if finetune_from_ema and resume_from:
        ema_file = os.path.join(resume_from, "ema.safetensors")
        if not os.path.exists(ema_file):
            error_msg = (
                f"❌ EMA FILE NOT FOUND: {ema_file}\n"
                f"   Checkpoint folder: {resume_from}\n"
                f"   Available files: {os.listdir(resume_from) if os.path.exists(resume_from) else 'N/A'}\n"
                "   \n"
                "   The checkpoint folder must contain ema.safetensors for EMA loading.\n"
                "   Either:\n"
                "   1. Use a different checkpoint that has EMA weights, or\n"
                "   2. Set --finetune_from_ema False to load main model weights instead"
            )
            logger.error(error_msg)
            raise FileNotFoundError(f"EMA weights file not found: {ema_file}")

    # Log final resume configuration
    logger.info("=== CHECKPOINT RESUME CONFIGURATION ===")
    logger.info(f"auto_resume: {training_args.auto_resume}")
    logger.info(f"resume_from: {resume_from}")
    logger.info(f"resume_model_only: {resume_model_only}")
    logger.info(f"finetune_from_ema: {finetune_from_ema}")
    logger.info(f"resume_model_optimizer: {training_args.resume_model_optimizer}")
    if finetune_from_ema:
        logger.info("🔄 Will load EMA weights (ema.safetensors) into main model for fine-tuning")
    elif resume_from:
        logger.info("🔄 Will load main model weights (model.safetensors)")
    else:
        logger.info("🆕 No checkpoint loading - starting fresh training")
    logger.info("=" * 40)

    # Set seed
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.visual_und:
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)
    else:
        vit_model = None
        vit_config = None

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors")
            if os.path.exists(os.path.join(model_args.model_path, "ae.safetensors")) else os.path.join(model_args.model_path, "vae_model.safetensors")
        )
    else:
        vae_model = None
        vae_config = None

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config,
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
        freeze_vae=training_args.freeze_vae,
    )

    # VAE integration based on freeze_vae
    if training_args.visual_gen and not training_args.freeze_vae:
        model = Bagel(
            language_model,
            vit_model if training_args.visual_und else None,
            config,
            vae_model=vae_model
        )
        vae_model = None
        logger.info("VAE integrated into Bagel model for training")
    else:
        model = Bagel(
            language_model,
            vit_model if training_args.visual_und else None,
            config,
            vae_model=None
        )
        logger.info("VAE kept external (frozen or not needed)")

    log_model_parameters(logger, model, language_model, 
                        vit_model if training_args.visual_und else None, 
                        vae_model if training_args.visual_gen else None, 
                        training_args)

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)

    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    if training_args.visual_gen:
        if hasattr(model, 'vae_model') and model.vae_model is not None:
            if training_args.freeze_vae:
                for param in model.vae_model.parameters():
                    param.requires_grad = False
                logger.info("Internal VAE parameters frozen")
            else:
                for param in model.vae_model.parameters():
                    param.requires_grad = True
                logger.info("Internal VAE parameters unfrozen for training")
        elif vae_model is not None:
            if training_args.freeze_vae:
                for param in vae_model.parameters():
                    param.requires_grad = False
                logger.info("External VAE parameters frozen")
            else:
                for param in vae_model.parameters():
                    param.requires_grad = True
                logger.info("External VAE parameters unfrozen for training")
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    logger.info("After applying freezing strategy:")
    log_model_parameters(logger, model, language_model,
                        vit_model if training_args.visual_und else None,
                        vae_model if training_args.visual_gen else None,
                        training_args)

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    
    
    ema_model = deepcopy(model)

    external_vae = vae_model if (training_args.visual_gen and training_args.freeze_vae) else None

    model, ema_model = FSDPCheckpoint.unified_load_checkpoint(
        resume_from, logger, model, ema_model,
        finetune_from_ema=finetune_from_ema,
        external_vae_model=external_vae
    )

    if training_args.visual_gen:
        if hasattr(model, 'vae_model') and model.vae_model is not None:
            vae_params = sum(p.numel() for p in model.vae_model.parameters())
            logger.info(f"Internal VAE: {vae_params/1e6:.2f}M parameters")
        elif external_vae is not None:
            vae_params = sum(p.numel() for p in external_vae.parameters())
            logger.info(f"External VAE: {vae_params/1e6:.2f}M parameters")
        else:
            logger.warning("Visual generation enabled but no VAE model found")
    ema_model = fsdp_ema_setup_with_ddp(ema_model, fsdp_config)

    fsdp_model = fsdp_wrapper_with_ddp(model, fsdp_config)

    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=grad_checkpoint_check_fn
    )

    if dist.get_rank() == 0:
        logger.info("FSDP model structure:")
        logger.info(f"{fsdp_model}")
        logger.info("Parameter status check:")
        for name, param in model.named_parameters():
            logger.info(f"{name}: requires_grad={param.requires_grad}")



    def create_param_groups(fsdp_model, training_args):
        assigned_params = set()

        param_groups = []


        if hasattr(fsdp_model, "vae_model") and fsdp_model.vae_model is not None:
            vae_params = list(fsdp_model.vae_model.parameters())
            param_groups.append({
                'params': vae_params,
                'lr': getattr(training_args, 'vae_lr', training_args.lr),
                'name': 'vae_model'
            })
            assigned_params.update(vae_params)

        if hasattr(fsdp_model, "vit_model") and fsdp_model.vit_model is not None:
            vit_params = list(fsdp_model.vit_model.parameters())
            param_groups.append({
                'params': vit_params,
                'lr': getattr(training_args, 'vit_lr', training_args.lr),
                'name': 'vit_model'
            })
            assigned_params.update(vit_params)

        if hasattr(fsdp_model, "language_model") and fsdp_model.language_model is not None:
            llm_params = list(fsdp_model.language_model.parameters())
            llm_params = [p for p in llm_params if p not in assigned_params]
            if llm_params:
                param_groups.append({
                    'params': llm_params,
                    'lr': getattr(training_args, 'llm_lr', training_args.lr),
                    'name': 'language_model'
                })
                assigned_params.update(llm_params)

        others_params = [
            p for p in fsdp_model.parameters() if p not in assigned_params
        ]
        if others_params:
            param_groups.append({
                'params': others_params,
                'lr': getattr(training_args, 'lr', 1e-5),
                'name': 'others'
            })

        return param_groups



    param_groups = create_param_groups(fsdp_model, training_args)

    if dist.get_rank() == 0:
        logger.info("=" * 60)
        logger.info("Optimizer parameter group configuration validation")
        logger.info("=" * 60)
        total_params = 0
        for i, group in enumerate(param_groups):
            group_name = group.get('name', f'group_{i}')
            group_lr = group.get('lr', 'default')
            param_count = len(group['params'])
            total_params += param_count
            logger.info(f"{group_name}: {param_count} parameters, learning_rate={group_lr}")
        logger.info(f"Total parameters managed by optimizer: {total_params}")
        logger.info("=" * 60)
    
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )


    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config,resume_model_optimizer = training_args.resume_model_optimizer
        )

    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
        logger.info("=" * 50)
        logger.info("Dataset configuration information:")
        logger.info(f"{dataset_meta}")
        logger.info("=" * 50)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )

    train_dataset.set_epoch(data_args.data_seed)

    if dist.get_rank() == 0:
        logger.info("Creating DataLoader...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=max(1, min(data_args.num_workers, 2)),
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=max(1, min(data_args.prefetch_factor, 2)),
        timeout=120,
        persistent_workers=True,
    )

    if dist.get_rank() == 0:
        logger.info(f"DataLoader creation completed - Worker processes: {max(1, min(data_args.num_workers, 2))}")

    torch.distributed.barrier()
    logger.info(f"Rank {dist.get_rank()} DataLoader initialization completed, ready to start training")

    def get_unified_vae_model():
        """Get unified VAE model reference"""
        if hasattr(fsdp_model.module, 'vae_model') and fsdp_model.module.vae_model is not None:
            return fsdp_model.module
        elif vae_model is not None:
            return vae_model
        else:
            return None
    
    if training_args.visual_gen:
        actual_vae = get_unified_vae_model()

        if hasattr(actual_vae, 'vae_model') and actual_vae.vae_model is not None:
            if training_args.freeze_vae:
                actual_vae.vae_model.eval()
            else:
                actual_vae.vae_model.train()
        elif actual_vae is not None:
            if hasattr(actual_vae, 'to') and callable(getattr(actual_vae, 'to')):
                actual_vae.to(device)

            if training_args.freeze_vae:
                actual_vae.eval()
            else:
                actual_vae.train()
    
    fsdp_model.train()
    ema_model.eval()

    start_time = time_second()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")

    actual_vae = get_unified_vae_model()
    if actual_vae is not None and hasattr(actual_vae, 'cuda'):
        actual_vae = actual_vae.cuda(device)

    for curr_step, data in enumerate(train_loader, start=train_step):
        if curr_step >= training_args.total_steps:
            break

        data = data.cuda(device).to_dict()

        data_check_passed = True
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any():
                    logger.error(f"Step {curr_step}: NaN detected in {key}")
                    data_check_passed = False

                if key == 'packed_timesteps':
                    if torch.isposinf(value).any():
                        logger.error(f"Step {curr_step}: Positive Inf detected in {key}")
                        data_check_passed = False
                else:
                    if torch.isinf(value).any():
                        logger.error(f"Step {curr_step}: Inf detected in {key}")
                        data_check_passed = False

        if not data_check_passed:
            logger.warning(f"Step {curr_step}: Data integrity check failed")
            
        torch.distributed.barrier()

        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)

        images_to_encode = data.pop('padded_images', None)
        if images_to_encode is not None and isinstance(images_to_encode, torch.Tensor) and images_to_encode.numel() > 0:
            assert images_to_encode.dim() == 4, f"Expect NCHW format, got {tuple(images_to_encode.shape)}"

            try:
                torch.cuda.synchronize()
                if training_args.freeze_vae:
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                            if hasattr(actual_vae, 'vae_encode'):
                                z_encoded = actual_vae.vae_encode(images_to_encode)
                            elif hasattr(actual_vae, 'encode'):
                                z_encoded = actual_vae.encode(images_to_encode)
                            else:
                                raise AttributeError(f"VAE model {type(actual_vae)} missing encode method")
                        data['padded_latent'] = z_encoded
                else:
                    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        if hasattr(actual_vae, 'vae_encode'):
                            z_encoded = actual_vae.vae_encode(images_to_encode)
                        elif hasattr(actual_vae, 'encode'):
                            z_encoded = actual_vae.encode(images_to_encode)
                        else:
                            raise AttributeError(f"VAE model {type(actual_vae)} missing encode method")
                    data['padded_latent'] = z_encoded
            except Exception as e:
                logger.error(f"VAE encoding failed: {e}")
                logger.error(f"actual_vae type: {type(actual_vae)}")
                logger.error(f"actual_vae available methods: {[m for m in dir(actual_vae) if not m.startswith('_')]}")
                raise

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            loss_dict = fsdp_model(**data)

        loss = 0

        ce = loss_dict["ce"]
        if ce is not None:
            total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)

            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens

            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        if training_args.visual_gen:
            mse = loss_dict["mse"]
            total_mse_tokens = torch.tensor(len(data.get('mse_loss_indexes', [])), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss += mse * training_args.mse_weight
        else:
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        optimizer.zero_grad()
   
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Step {curr_step}: Abnormal loss value: {loss}")
            torch.distributed.barrier()
        
        loss.backward()

        total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)

        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            torch.cuda.synchronize()
            end_time = time_second()
            steps_per_sec = training_args.log_every / (end_time - start_time)
            message = f"(step={curr_step:07d}) "
            log = {}

            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        value = value.mean()

                    if value.device != device:
                        value = value.to(device)
                    
                    avg_loss = torch.tensor(value.item(), device=device)
                else:
                    avg_loss = torch.tensor(float(value), device=device)
                
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                log[key] = avg_loss

            message += f"Train Steps/Sec: {steps_per_sec:.2f}, "
            logger.info(message)

            for i, group in enumerate(optimizer.param_groups):
                group_name = group.get('name', f'group_{i}')
                log[f'lr_{group_name}'] = group['lr']
            log['lr'] = optimizer.param_groups[0]['lr']
            log['total_mse_tokens'] = total_mse_tokens.item()
            log['total_ce_tokens'] = total_ce_tokens.item()
            log['total_norm'] = total_norm.item()
            log['total_samples'] = total_samples.item()
            log['training_mode'] = 'standard'

            if dist.get_rank() == 0 and writer is not None:
                for key, value in log.items():
                    if isinstance(value, (int, float, torch.Tensor)):
                        scalar_value = value.item() if isinstance(value, torch.Tensor) else value
                        writer.add_scalar(f'train/{key}', scalar_value, curr_step)
            start_time = time_second()

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            logger.info("Saving checkpoint...")
            torch.cuda.empty_cache()
            
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            dist.gather_object(data_status, gather_list, dst=0)

            actual_save_path = FSDPCheckpoint.unified_save_checkpoint(
                save_path=os.path.join(training_args.checkpoint_dir, f"{curr_step:07d}"),
                train_steps=curr_step,
                model=fsdp_model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                data_status=gather_list,
                logger=logger,
                fsdp_config=fsdp_config,
                tokenizer=tokenizer,
                vae_model=vae_model if training_args.visual_gen else None,
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
            )
            
            if dist.get_rank() == 0:
                all_dirs = glob.glob(os.path.join(training_args.checkpoint_dir, "*"))
                ckpt_dirs = []
                
                for d in all_dirs:
                    if os.path.isdir(d):
                        dirname = os.path.basename(d)
                        try:
                            int(dirname)
                            ckpt_dirs.append(d)
                        except ValueError:
                            continue
                
                ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(os.path.basename(x)))
                
                while len(ckpt_dirs) > training_args.max_checkpoints:
                    oldest = ckpt_dirs.pop(0)
                    shutil.rmtree(oldest)
                    logger.info(f"Removed old checkpoint: {oldest}")

    if dist.get_rank() == 0:
        logger.info("=" * 50)
        logger.info(f"Training completed: {curr_step} steps")
        logger.info("=" * 50)
        if writer is not None:
            writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
