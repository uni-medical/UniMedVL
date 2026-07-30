# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import os
import time
import json
import shutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

from modeling.unimedvl.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.autoencoder import AutoEncoder, Decoder, Encoder, DiagonalGaussian
from modeling.unimedvl.qwen2_navit import (
    Qwen2DecoderLayer,
    Qwen2MoEDecoderLayer,
    Qwen2MoTDecoderLayer,

)
from modeling.unimedvl.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer


_logger = logging.getLogger(__name__)


def dataclass_to_dict(obj):
    """Convert dataclass object to serializable dictionary"""
    if obj is None:
        return None
    if hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    result[key] = value
                else:
                    result[key] = str(value)
        return result
    else:
        return str(obj)


def get_clean_state_dict(model_or_state_dict, component_name=None):
    """
    Get cleaned state dictionary with all wrapper prefixes removed
    
    Args:
        model_or_state_dict: Model instance or state dictionary
        component_name: Component name (e.g., 'vae_model', 'vit_model')
    
    Returns:
        clean_dict: Cleaned state dictionary  
        metadata: Wrapper information metadata
    """
    if hasattr(model_or_state_dict, 'state_dict'):
        state_dict = model_or_state_dict.state_dict()
    else:
        state_dict = model_or_state_dict
    
    clean_dict = {}
    metadata = {
        'has_fsdp': False,
        'has_ddp': False,
        'component_prefix': component_name,
        'original_keys_sample': list(state_dict.keys())[:5],
        'total_keys': len(state_dict),
        'ddp_layers_detected': 0
    }
    
    for key, value in state_dict.items():
        clean_key = key
        
        if 'fsdp' in key.lower():
            metadata['has_fsdp'] = True
        if '.module.' in key:
            metadata['has_ddp'] = True
            metadata['ddp_layers_detected'] += key.count('.module.')
            
        if component_name:
            if key.startswith(f'{component_name}.'):
                clean_key = key[len(f'{component_name}.'):]
            else:
                continue
        
        while '.module.' in clean_key:
            clean_key = clean_key.replace('.module.', '.')
        
        if clean_key.startswith('module.'):
            clean_key = clean_key[len('module.'):]
                
        clean_dict[clean_key] = value
    
    return clean_dict, metadata


def detect_model_wrapping(model, component_name=None):
    """
    Detect model wrapping status
    
    Args:
        model: Model to be detected
        component_name: Component name
    
    Returns:
        wrapping_info: Wrapping information dictionary
        actual_model: Actual model after removing FSDP wrapping
    """
    wrapping_info = {
        'has_fsdp': False,
        'has_ddp': False,
        'component_exists': False,
        'component_has_ddp': False,
        'component_path': None,
        'fsdp_module_path': None
    }
    
    actual_model = model
    if hasattr(model, '_fsdp_wrapped_module'):
        wrapping_info['has_fsdp'] = True
        wrapping_info['fsdp_module_path'] = '_fsdp_wrapped_module'
        actual_model = model._fsdp_wrapped_module
    if hasattr(model, 'module'):
        wrapping_info['has_ddp'] = True
        actual_model = model.module
    
    if component_name and hasattr(actual_model, component_name):
        wrapping_info['component_exists'] = True
        wrapping_info['component_path'] = component_name
        component = getattr(actual_model, component_name)
        
        if hasattr(component, 'module'):
            wrapping_info['component_has_ddp'] = True
            
    return wrapping_info, actual_model


def apply_component_prefix(state_dict, component_name, target_has_ddp=False):
    """
    Add component prefix and DDP wrapping to state dictionary (if needed)
    
    Args:
        state_dict: Cleaned state dictionary
        component_name: Component name
        target_has_ddp: Whether target model has DDP wrapping
    
    Returns:
        prefixed_dict: State dictionary with prefix added
    """
    prefixed_dict = {}
    
    for key, value in state_dict.items():
        if component_name:
            if target_has_ddp:
                prefixed_key = f'{component_name}.module.{key}'
            else:
                prefixed_key = f'{component_name}.{key}'
        else:
            if target_has_ddp:
                prefixed_key = f'module.{key}'
            else:
                prefixed_key = key
                
        prefixed_dict[prefixed_key] = value
    
    return prefixed_dict


def save_component_weights(full_state_dict, save_path, component_name, logger):
    """
    Intelligently save component weights, ensuring consistent format
    
    Args:
        full_state_dict: Complete model state dictionary
        save_path: Save path
        component_name: Component name (e.g., 'vae_model', 'vit_model')
        logger: Logger
    
    Returns:
        bool: Whether save was successful
    """
    logger.info(f"Saving {component_name} component...")
    
    component_keys = [k for k in full_state_dict.keys() if component_name in k]
    logger.debug(f"Found {len(component_keys)} keys containing '{component_name}'")
    
    clean_weights, metadata = get_clean_state_dict(full_state_dict, component_name)
    
    if clean_weights:
        component_file = os.path.join(save_path, f"{component_name}.safetensors")
        save_file(clean_weights, component_file)
        logger.info(f"Successfully saved {component_name} weights: {len(clean_weights)} keys")
        
        return True
    else:
        logger.warning(f"{component_name} weights not found")
        return False


def load_component_weights(model, component_name, checkpoint_path, logger):
    """
    Intelligently load component weights, automatically adapting to wrapping format
    
    Args:
        model: Target model
        component_name: Component name
        checkpoint_path: Checkpoint path
        logger: Logger
    
    Returns:
        bool: Whether loading was successful
    """
    logger.info(f"Loading {component_name} component...")
    
    component_file = os.path.join(checkpoint_path, f"{component_name}.safetensors")
    if not os.path.exists(component_file):
        if "vae_model" in component_file:
            component_file = os.path.join(checkpoint_path, f"ae.safetensors")
            if not os.path.exists(component_file):
                logger.warning(f"Weight file not found: {component_file}")
                return False
        else:
            logger.warning(f"Weight file not found: {component_file}")
            return False
    
    clean_weights = load_file(component_file, device="cpu")
    logger.debug(f"Loaded {len(clean_weights)} weight keys from file")
    
    wrapping_info, actual_model = detect_model_wrapping(model, component_name)
    
    if not wrapping_info['component_exists']:
        logger.error(f"{component_name} component not found in target model!")
        return False
    
    logger.debug(f"Target DDP wrapping: {wrapping_info['component_has_ddp']}")
    
    target_weights = apply_component_prefix(
        clean_weights, 
        component_name,
        target_has_ddp=wrapping_info['component_has_ddp']
    )
    
    loading_result = model.load_state_dict(target_weights, strict=False)
    missing_keys = loading_result.missing_keys
    unexpected_keys = loading_result.unexpected_keys
    
    component_missing = [k for k in missing_keys if component_name in k]
    component_unexpected = [k for k in unexpected_keys if component_name in k]
    
    if component_missing:
        logger.warning(f"{component_name} missing keys: {len(component_missing)}")
        logger.debug(f"First 5 missing keys: {component_missing[:5]}")
    
    if component_unexpected:
        logger.debug(f"{component_name} unexpected keys: {len(component_unexpected)}")
    
    success_count = len(clean_weights) - len(component_missing)
    success_rate = (success_count / len(clean_weights)) * 100 if clean_weights else 0
    
    if success_rate >= 95:
        logger.info(f"✅ [COMPONENT LOAD DEBUG] {component_name} loading success: {success_rate:.1f}% ({success_count}/{len(clean_weights)} keys)")
    elif success_rate >= 80:
        logger.warning(f"⚠️ [COMPONENT LOAD DEBUG] {component_name} loading partial success: {success_rate:.1f}% ({success_count}/{len(clean_weights)} keys)")
    else:
        logger.error(f"❌ [COMPONENT LOAD DEBUG] {component_name} loading low success: {success_rate:.1f}% ({success_count}/{len(clean_weights)} keys)")
    
    
    return len(component_missing) == 0



def migrate_old_checkpoint(checkpoint_path, logger):
    """
    Migrate old format checkpoint to new format
    """
    migrated = False
    
    # Migrate vit.safetensors -> vit_model.safetensors
    old_vit = os.path.join(checkpoint_path, "vit.safetensors")
    new_vit = os.path.join(checkpoint_path, "vit_model.safetensors")
    
    if os.path.exists(old_vit) and not os.path.exists(new_vit):
        shutil.copy2(old_vit, new_vit)
        logger.info("ViT weight file migration: vit.safetensors -> vit_model.safetensors")
        migrated = True
    
    if migrated:
        logger.info("Old format checkpoint migration completed")
    
    return migrated


def validate_checkpoint_integrity(checkpoint_path, expected_components, logger):
    """
    Validate checkpoint integrity
    """
    missing_components = []
    found_components = []
    
    for component in expected_components:
        # All components use safetensors format
        component_file = os.path.join(checkpoint_path, f"{component}.safetensors")
            
        if not os.path.exists(component_file):
            missing_components.append(component)
        else:
            found_components.append(component)
    
    if missing_components:
        logger.warning(f"Missing components: {missing_components}")
    
    if found_components:
        logger.info(f"Found components: {found_components}")
    
    return len(missing_components) == 0



class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard

def fsdp_wrapper_with_ddp(original_model, fsdp_config, ignored_modules=None):
    """Mixed FSDP+DDP wrapper: only wraps vae_model"""
    if ignored_modules is None:
        ignored_modules = []
    
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    
    # Ensure model is on the correct GPU
    original_model = original_model.cuda(device_id)
    
    # Collect modules that need DDP wrapping
    ddp_wrapped_modules = []

    if hasattr(original_model, 'vae_model') and original_model.vae_model is not None:
        _logger.debug(f"[Rank {rank}] Found vae_model, preparing for DDP wrapping")
        try:
            original_model.vae_model = original_model.vae_model.cuda(device_id)
            original_model.vae_model = DDP(
                original_model.vae_model,
                device_ids=[device_id],
                output_device=device_id,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            ddp_wrapped_modules.append(original_model.vae_model)
            _logger.info(f"[Rank {rank}] vae_model wrapped with DDP")
        except Exception as e:
            _logger.warning(f"[Rank {rank}] vae_model DDP wrapping failed: {e}")

    if hasattr(original_model, 'vae_batch_norm') and original_model.vae_batch_norm is not None:
        try:
            original_model.vae_batch_norm = original_model.vae_batch_norm.cuda(device_id).float()
            ignored_modules.append(original_model.vae_batch_norm)
            _logger.debug(f"[Rank {rank}] vae_batch_norm added to ignored_modules")
        except Exception as e:
            _logger.warning(f"[Rank {rank}] vae_batch_norm processing failed: {e}")
    
    batch_norm_attrs = ['batch_norm', 'norm', 'layer_norm', 'group_norm']  
    for attr_name in batch_norm_attrs:
        if hasattr(original_model, attr_name):
            norm_layer = getattr(original_model, attr_name)
            if norm_layer is not None and isinstance(norm_layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
                try:
                    norm_layer = norm_layer.cuda(device_id).float()
                    setattr(original_model, attr_name, norm_layer)
                    ignored_modules.append(norm_layer)
                    _logger.debug(f"[Rank {rank}] {attr_name} added to ignored_modules")
                except Exception as e:
                    _logger.warning(f"[Rank {rank}] {attr_name} processing failed: {e}")
    
    ignored_modules.extend(ddp_wrapped_modules)
    _logger.debug(f"[Rank {rank}] {len(ddp_wrapped_modules)} modules wrapped with DDP")
    
    if ddp_wrapped_modules:
        for ddp_module in ddp_wrapped_modules:
            if hasattr(ddp_module, 'module'):
                try:
                    ddp_module.module = ddp_module.module.to(dtype=torch.bfloat16)
                    _logger.debug(f"[Rank {rank}] DDP-wrapped VAE converted to bfloat16")
                except Exception as e:
                    _logger.warning(f"[Rank {rank}] VAE precision conversion failed: {e}")
    
    if hasattr(original_model, 'vae_batch_norm') and original_model.vae_batch_norm is not None:
        try:
            original_model.vae_batch_norm._forward_pre_hooks.clear()
            _logger.debug(f"[Rank {rank}] BatchNorm configured for autocast")
        except Exception as e:
            _logger.warning(f"[Rank {rank}] BatchNorm autocast configuration failed: {e}")
    
    device_mesh = None
    if getattr(fsdp_config, "sharding_strategy", None) == 'HYBRID_SHARD':
        try:
            device_mesh = init_device_mesh(
                "cuda", 
                mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
                mesh_dim_names=("replicate", "shard")
            )
            _logger.debug(f"[Rank {rank}] Device mesh initialized")
        except Exception as e:
            _logger.warning(f"[Rank {rank}] Device mesh initialization failed: {e}")
            device_mesh = None
    
    transformer_layer_cls = {
        Qwen2DecoderLayer,
        Qwen2MoEDecoderLayer,
        Qwen2MoTDecoderLayer,
        SiglipEncoderLayer,
        SiglipVisionTransformer,
        MLPconnector,
        TimestepEmbedder,
        PositionEmbedding,
    }
    
    _logger.debug(f"[Rank {rank}] FSDP will ignore {len(ignored_modules)} modules")

    try:
        fsdp_model = FSDP(
            original_model,
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=transformer_layer_cls,
            ),
            ignored_modules=ignored_modules,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            device_id=device_id,
            sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
            backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
            cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
            device_mesh=device_mesh,
            use_orig_params=False
        )
        _logger.info(f"[Rank {rank}] FSDP+DDP hybrid model created")
        return fsdp_model
    except Exception as e:
        _logger.error(f"[Rank {rank}] FSDP model creation failed: {e}")
        raise



def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[]):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    
    transformer_layer_cls = {
        Qwen2DecoderLayer,
        Qwen2MoEDecoderLayer,
        Qwen2MoTDecoderLayer,
        SiglipEncoderLayer,
        SiglipVisionTransformer,
        MLPconnector,
        TimestepEmbedder,
        PositionEmbedding,
    }

    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_cls,
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        use_orig_params=False
    )


class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scheduler, 
        data_status,
        logger, 
        fsdp_config,
        # New parameters for saving complete inference files (backward compatible)
        tokenizer=None,
        vae_model=None,
        model_args=None,
        data_args=None,
        training_args=None,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        
        if torch.cuda.is_available() and logger is not None and logger.isEnabledFor(logging.DEBUG):
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            logger.debug(f"Memory before save: {allocated_before:.2f}GB allocated, {reserved_before:.2f}GB reserved")
        
        logger.info(f"Saving checkpoint to {save_path}.")

        main_model_state_dict = None
        
        if ema_model is not None:
            torch.cuda.empty_cache()
            
            try:
                with FSDP.state_dict_type(
                    ema_model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
                ):
                    torch.cuda.synchronize()
                    ema_state_dict = ema_model.state_dict()
                    
                    if dist.get_rank() == 0:
                        save_file(ema_state_dict, os.path.join(save_path, "ema.safetensors"))
                        logger.info("Saved EMA model weights")
                        del ema_state_dict
                        torch.cuda.empty_cache()
                        
            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"CUDA OOM while saving EMA model: {e}")
                logger.warning("Using LOCAL_STATE_DICT fallback")
                
                torch.cuda.empty_cache()
                
                with FSDP.state_dict_type(ema_model, StateDictType.LOCAL_STATE_DICT):
                    local_ema_state_dict = ema_model.state_dict()
                    if dist.get_rank() == 0:
                        save_file(local_ema_state_dict, os.path.join(save_path, "ema_local.safetensors"))
                        del local_ema_state_dict
                        torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        
        try:
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
            ):
                torch.cuda.synchronize()
                main_model_state_dict = model.state_dict()
                
                if dist.get_rank() == 0:
                    logger.info("Saving main model...")
                    logger.debug(f"Main model state dict contains {len(main_model_state_dict)} keys")
                    
                    clean_main_state_dict, main_metadata = get_clean_state_dict(main_model_state_dict)
                    logger.debug(f"Wrapper cleaning: DDP={main_metadata['has_ddp']}, FSDP={main_metadata['has_fsdp']}")
                    
                    model_file = os.path.join(save_path, "model.safetensors")
                    save_file(clean_main_state_dict, model_file)
                    logger.info(f"Saved main model: {len(clean_main_state_dict)} keys")
                    
                    logger.debug("Starting component-wise saving...")
                    
                    vit_saved = save_component_weights(main_model_state_dict, save_path, "vit_model", logger)
                    vae_saved = save_component_weights(main_model_state_dict, save_path, "vae_model", logger)
                    
                    llm_saved = save_component_weights(main_model_state_dict, save_path, "language_model", logger)
                    
                    if not vae_saved and vae_model is not None:
                        try:
                            logger.info("💾 [VAE SAVE DEBUG] Saving external VAE weights to checkpoint...")
                            if hasattr(vae_model, 'state_dict'):
                                external_vae_state_dict = vae_model.state_dict()
                                if external_vae_state_dict:
                                    clean_vae_weights, metadata = get_clean_state_dict(external_vae_state_dict)
                                    
                                   
                                    vae_model_file = os.path.join(save_path, "vae_model.safetensors")
                                    save_file(clean_vae_weights, vae_model_file)
                                    
                                    logger.info(f"✅ [VAE SAVE DEBUG] External VAE saved successfully: {len(clean_vae_weights)} keys")
                                    vae_saved = True
                                    
                                    del external_vae_state_dict, clean_vae_weights
                        except Exception as e:
                            logger.error(f"❌ [VAE SAVE DEBUG] Failed to save external VAE: {e}")
                    
                    save_summary = {
                        'vit_model': vit_saved,
                        'vae_model': vae_saved,
                        'language_model': llm_saved,
                        'total_keys': len(main_model_state_dict)
                    }
                    logger.debug(f"Component save summary: {save_summary}")
                    
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"CUDA OOM while saving main model: {e}")
            logger.warning("Skipping main model save to prevent crash")

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_save_path = os.path.join(
                save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                torch.save(optimizer.state_dict(), optimizer_save_path)
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                if dist.get_rank() < fsdp_config.num_shard:
                    torch.save(optimizer.state_dict(), optimizer_save_path)
            else:
                raise NotImplementedError

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if dist.get_rank() == 0 and data_status is not None:
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        if dist.get_rank() == 0:
            FSDPCheckpoint._save_inference_files(
                save_path, model, tokenizer, vae_model, model_args, data_args, training_args, logger
            )

        if torch.cuda.is_available() and logger is not None and logger.isEnabledFor(logging.DEBUG):
            torch.cuda.empty_cache()
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            reserved_after = torch.cuda.memory_reserved() / 1024**3
            logger.debug(f"Memory after save: {allocated_after:.2f}GB allocated, {reserved_after:.2f}GB reserved")
        
        dist.barrier()
        logger.info(f"Checkpoint saved successfully to {save_path}")
        return

    @staticmethod
    def _save_inference_files(save_path, model, tokenizer, vae_model, model_args, data_args, training_args, logger):
        """Save complete file structure needed for inference"""
        try:
            logger.info("Saving inference files...")
            
            if hasattr(model, 'language_model') and hasattr(model.language_model, 'config'):
                llm_config = model.language_model.config
                llm_config_dict = llm_config.to_dict()
                with open(os.path.join(save_path, "llm_config.json"), 'w') as f:
                    json.dump(llm_config_dict, f, indent=2)
                logger.debug("Saved llm_config.json")

            if hasattr(model, 'vit_model') and hasattr(model.vit_model, 'config'):
                vit_config = model.vit_model.config
                vit_config_dict = vit_config.to_dict()
                with open(os.path.join(save_path, "vit_config.json"), 'w') as f:
                    json.dump(vit_config_dict, f, indent=2)
                logger.debug("Saved vit_config.json")

            config_dict = {}

            if hasattr(model, 'config'):
                try:
                    bagel_config_dict = model.config.to_dict() if hasattr(model.config, 'to_dict') else {}
                    
                    def fix_vae_config(config_dict):
                        """Fix vae_config serialization issue"""
                        if isinstance(config_dict, dict):
                            for key, value in config_dict.items():
                                if hasattr(value, '__dataclass_fields__') and type(value).__name__ == 'AutoEncoderParams':
                                    config_dict[key] = {
                                        'resolution': value.resolution,
                                        'in_channels': value.in_channels,
                                        'downsample': value.downsample,
                                        'ch': value.ch,
                                        'out_ch': value.out_ch,
                                        'ch_mult': list(value.ch_mult),
                                        'num_res_blocks': value.num_res_blocks,
                                        'z_channels': value.z_channels,
                                        'scale_factor': value.scale_factor,
                                        'shift_factor': value.shift_factor
                                    }
                                elif isinstance(value, dict):
                                    fix_vae_config(value)
                        return config_dict
                    
                    config_dict["model_config"] = fix_vae_config(bagel_config_dict)
                    logger.debug("Serialized Bagel config")
                    
                except Exception as e:
                    logger.warning(f"Unable to serialize Bagel config: {e}")
                    config_dict["model_config"] = {
                        "visual_gen": getattr(model.config, 'visual_gen', True),
                        "visual_und": getattr(model.config, 'visual_und', True),
                        "latent_patch_size": getattr(model.config, 'latent_patch_size', 2),
                        "max_latent_size": getattr(model.config, 'max_latent_size', 64),
                        "timestep_shift": getattr(model.config, 'timestep_shift', 1.0),
                        "error": f"Configuration serialization failed: {str(e)}"
                    }
            
            config_dict["model_args"] = dataclass_to_dict(model_args)
            config_dict["data_args"] = dataclass_to_dict(data_args)
            config_dict["training_args"] = dataclass_to_dict(training_args)
            
            config_dict["meta"] = {
                "config_version": "2.0",
                "saved_timestamp": datetime.now().isoformat(),
                "pytorch_version": torch.__version__ if 'torch' in globals() else "unknown",
                "transformers_version": "4.44.0",
            }
            
            config_json_path = os.path.join(save_path, "config.json")
            try:
                with open(config_json_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                logger.debug("Saved config.json")
            except Exception as e:
                logger.error(f"Failed to save config.json: {e}")

            generation_config = {
                "do_sample": True,
                "max_new_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "pad_token_id": 0,
                "eos_token_id": 151645,
                "bos_token_id": 151644,
            }
            
            if tokenizer:
                if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                    generation_config["pad_token_id"] = tokenizer.pad_token_id
                if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                    generation_config["eos_token_id"] = tokenizer.eos_token_id
                if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                    generation_config["bos_token_id"] = tokenizer.bos_token_id
            
            with open(os.path.join(save_path, "generation_config.json"), 'w') as f:
                json.dump(generation_config, f, indent=2)
            logger.debug("Saved generation_config.json")

            all_training_args = {
                "model_args": dataclass_to_dict(model_args),
                "data_args": dataclass_to_dict(data_args), 
                "training_args": dataclass_to_dict(training_args),
                "saved_timestamp": datetime.now().isoformat(),
                "train_steps": train_steps,
            }
            
            with open(os.path.join(save_path, "training_args.json"), 'w') as f:
                json.dump(all_training_args, f, indent=2, ensure_ascii=False)
            logger.debug("Saved training_args.json")

            if tokenizer:
                try:
                    tokenizer.save_pretrained(save_path)
                    logger.debug("Saved tokenizer files")
                except Exception as e:
                    logger.warning(f"Failed to save tokenizer: {e}")

            vae_saved = False
            
            if hasattr(model, 'vae_model') and model.vae_model is not None:
                try:
                    logger.debug("Saving VAE weights from integrated Bagel model...")
                    
                    with FSDP.state_dict_type(
                        model,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
                    ):
                        torch.cuda.synchronize()
                        full_state_dict = model.state_dict()
                        
                        vae_saved = save_component_weights(full_state_dict, save_path, "vae_model", logger)
                        
                        del full_state_dict
                        
                except Exception as e:
                    logger.warning(f"Failed to save integrated VAE weights: {e}")
            
            if not vae_saved and vae_model is not None:
                try:
                    logger.debug("Saving VAE weights from external model...")
                    if hasattr(vae_model, 'state_dict'):
                        external_vae_state_dict = vae_model.state_dict()
                        if external_vae_state_dict:
                            clean_vae_weights, metadata = get_clean_state_dict(external_vae_state_dict)
                            
                            vae_model_path = os.path.join(save_path, "vae_model.safetensors")
                            save_file(clean_vae_weights, vae_model_path)
                            
                            logger.debug(f"Saved external VAE weights: {len(clean_vae_weights)} keys")
                            vae_saved = True
                            del external_vae_state_dict, clean_vae_weights
                except Exception as e:
                    logger.warning(f"Failed to save external VAE weights: {e}")

            model_index = {
                "metadata": {
                    "total_size": 0,
                    "format": "safetensors",
                    "unified_format_version": "1.0"
                },
                "weight_map": {
                    "model": "model.safetensors",
                    "ema": "ema.safetensors"
                }
            }
            
            if vae_saved:
                model_index["weight_map"]["vae_model"] = "vae_model.safetensors"
            
            vit_model_path = os.path.join(save_path, "vit_model.safetensors")
            if os.path.exists(vit_model_path):
                model_index["weight_map"]["vit_model"] = "vit_model.safetensors"
            
            vit_path = os.path.join(save_path, "vit.safetensors")
            if os.path.exists(vit_path):
                model_index["weight_map"]["vit"] = "vit.safetensors"
            
            llm_path = os.path.join(save_path, "language_model.safetensors")
            if os.path.exists(llm_path):
                model_index["weight_map"]["language_model"] = "language_model.safetensors"
            
            with open(os.path.join(save_path, "model.safetensors.index.json"), 'w') as f:
                json.dump(model_index, f, indent=2)
            logger.debug("Saved model.safetensors.index.json")

            logger.info("Saved all inference files")

        except Exception as e:
            logger.error(f"Error saving inference files: {e}")

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, finetune_from_ema=False, external_vae_model=None):
        """Use unified logic to load checkpoint, automatically adapt to wrapper format"""
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}")
            
            # Migrate old format checkpoint if needed
            migrated = migrate_old_checkpoint(resume_from, logger)
            
            # Determine which model file to load
            if finetune_from_ema:
                model_file = os.path.join(resume_from, "ema.safetensors")
                logger.info("Loading EMA weights into main model for fine-tuning")
            else:
                model_file = os.path.join(resume_from, "model.safetensors")
                logger.info("Loading standard model weights")
            
            full_model_loaded = False
            if os.path.exists(model_file):
                model_state_dict = load_file(model_file, device="cpu")
                
                # Remove position embedding keys (will be reinitialized)
                pos_embed_keys = [
                    'latent_pos_embed.pos_embed', 'vit_pos_embed.pos_embed',
                    'bagel.latent_pos_embed.pos_embed', 'bagel.vit_pos_embed.pos_embed'
                ]
                for key in pos_embed_keys:
                    model_state_dict.pop(key, None)
                
                # Check compatibility
                model_keys = set(model.state_dict().keys())
                ckpt_keys = set(model_state_dict.keys())
                missing_keys = model_keys - ckpt_keys
                
                compatibility_ratio = len(missing_keys) / len(model_keys) if model_keys else 0
                if compatibility_ratio > 0.3:
                    logger.error(f"Serious compatibility issue! Missing {compatibility_ratio:.1%} of model keys")
                    logger.error(f"First 10 missing keys: {list(missing_keys)[:10]}")
                elif compatibility_ratio > 0.1:
                    logger.warning(f"Compatibility warning: Missing {compatibility_ratio:.1%} of keys")
                    logger.warning(f"First 5 missing keys: {list(missing_keys)[:5]}")
                
                # Load to model
                loading_result = model.load_state_dict(model_state_dict, strict=False)
                
                final_missing = len(loading_result.missing_keys)
                final_unexpected = len(loading_result.unexpected_keys)
                
                if final_missing > 0:
                    logger.warning(f"Model loading: {final_missing} missing keys, {final_unexpected} unexpected keys")
                    if final_missing <= 10:
                        logger.warning(f"Missing keys: {loading_result.missing_keys}")
                    else:
                        logger.warning(f"First 10 missing keys: {loading_result.missing_keys[:10]}")
                else:
                    logger.info(f"Model loaded successfully ({len(model_state_dict)} keys)")
                
                del model_state_dict
                full_model_loaded = True
            else:
                logger.error(f"Model file does not exist: {model_file}")
            
            # Component-wise loading (for VAE if not frozen)
            components_loaded = {}
            
            # Check if VAE is frozen (external VAE doesn't need to be loaded from checkpoint)
            vae_is_frozen = False
            training_args_file = os.path.join(resume_from, "training_args.json")
            if os.path.exists(training_args_file):
                try:
                    import json
                    with open(training_args_file, 'r') as f:
                        training_args_data = json.load(f)
                        if isinstance(training_args_data, dict) and 'training_args' in training_args_data:
                            freeze_vae = training_args_data['training_args'].get('freeze_vae', None)
                            if freeze_vae is True:
                                vae_is_frozen = True
                                logger.info("Detected frozen VAE, skipping VAE component loading")
                except Exception as e:
                    pass
            
            # Only load VAE component if it's not frozen
            expected_components = [] if vae_is_frozen else ['vae_model']
            
            # Load components
            for component in expected_components:
                success = load_component_weights(model, component, resume_from, logger)
                components_loaded[component] = success
            
            if components_loaded:
                successful = [k for k, v in components_loaded.items() if v]
                failed = [k for k, v in components_loaded.items() if not v]
                if failed:
                    logger.warning(f"Component loading: success={successful}, failed={failed}")
            
            # Handle external VAE model
            if external_vae_model is not None:
                vae_file = os.path.join(resume_from, "vae_model.safetensors")
                if not os.path.exists(vae_file):
                    vae_file = os.path.join(resume_from, "ae.safetensors")
                
                if os.path.exists(vae_file):
                    vae_weights = load_file(vae_file, device="cpu")
                    clean_vae_weights, metadata = get_clean_state_dict(vae_weights)
                    
                    try:
                        vae_result = external_vae_model.load_state_dict(clean_vae_weights, strict=False)
                        if len(vae_result.missing_keys) > 0:
                            logger.warning(f"External VAE loading: {len(vae_result.missing_keys)} missing keys")
                        else:
                            logger.info("External VAE loaded successfully")
                    except Exception as e:
                        logger.warning(f"Failed to load external VAE: {e}")
                    
                    del vae_weights, clean_vae_weights

            # Handle EMA model
            if ema_model is not None:
                ema_file = os.path.join(resume_from, "ema.safetensors")
                if not os.path.exists(ema_file):
                    ema_file = os.path.join(resume_from, "model.safetensors")
                    logger.info("EMA file not found, using main model weights")
                
                if os.path.exists(ema_file):
                    ema_state_dict = load_file(ema_file, device="cpu")
                    
                    pos_embed_keys = [
                        'latent_pos_embed.pos_embed', 'vit_pos_embed.pos_embed',
                        'bagel.latent_pos_embed.pos_embed', 'bagel.vit_pos_embed.pos_embed'
                    ]
                    for key in pos_embed_keys:
                        ema_state_dict.pop(key, None)
                    
                    ema_result = ema_model.load_state_dict(ema_state_dict, strict=False)
                    success_rate = ((len(ema_state_dict) - len(ema_result.missing_keys)) / len(ema_state_dict)) * 100 if ema_state_dict else 0
                    
                    if len(ema_result.missing_keys) == 0:
                        logger.info(f"EMA model loaded successfully ({len(ema_state_dict)} keys)")
                    elif success_rate >= 95:
                        logger.info(f"EMA model loaded with {success_rate:.1f}% success rate")
                    else:
                        logger.warning(f"EMA model loaded with issues: {success_rate:.1f}% success rate, {len(ema_result.missing_keys)} missing keys")
                    
                    del ema_state_dict
                else:
                    logger.warning("EMA model file does not exist")
            
        else:
            logger.info("Training from scratch.")
        
        return model, ema_model


    @staticmethod
    def unified_save_checkpoint(save_path, train_steps, model, ema_model, optimizer, scheduler,
                               data_status, logger, fsdp_config, **kwargs):
        """
        Unified checkpoint save interface, integrating all improvements

        Args:
            save_path: Can be one of the following two formats:
                      1. Directory path: '/path/to/checkpoints'
                      2. Complete path: '/path/to/checkpoints/0002000'
            train_steps: Training steps

        Returns:
            actual_save_path: Actual complete save path
        """
        logger.info(f"Using unified checkpoint saving to {save_path}")
        
        # Smart path handling: supports both directory path and complete path formats
        normalized_path = os.path.normpath(os.path.abspath(save_path))
        step_pattern = f"{train_steps:07d}"
        path_basename = os.path.basename(normalized_path)
        
        if path_basename == step_pattern:
            # Input is complete path format
            ckpt_dir = os.path.dirname(normalized_path)
            actual_save_path = normalized_path
        elif path_basename.isdigit() and len(path_basename) == 7:
            # Input is complete path but step number doesn't match
            logger.warning(f"Path step mismatch: '{path_basename}' vs {train_steps}, correcting...")
            ckpt_dir = os.path.dirname(normalized_path)
            actual_save_path = os.path.join(ckpt_dir, step_pattern)
        else:
            # Input is directory path format
            ckpt_dir = normalized_path
            actual_save_path = os.path.join(ckpt_dir, step_pattern)
        
        logger.info(f"Checkpoint directory: {ckpt_dir}")
        logger.info(f"Actual save path: {actual_save_path}")
        
        # Ensure directory exists
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Validate expected components
        expected_components = []
        if hasattr(model, 'vae_model') or kwargs.get('vae_model') is not None:
            expected_components.append('vae_model')
        if hasattr(model, 'vit_model'):
            expected_components.append('vit_model')  
        if hasattr(model, 'language_model'):
            expected_components.append('language_model')
        
        # Call original save function (pass directory path)
        FSDPCheckpoint.fsdp_save_ckpt(
            ckpt_dir, train_steps, model, ema_model, 
            optimizer, scheduler, data_status, logger, fsdp_config, **kwargs
        )

        # Validate save results
        integrity_ok = validate_checkpoint_integrity(actual_save_path, expected_components, logger)
        
        if integrity_ok:
            logger.info(f"Checkpoint integrity validation passed")
        else:
            logger.warning(f"Checkpoint integrity validation failed, but save completed")
        
        return actual_save_path

    @staticmethod
    def unified_load_checkpoint(resume_from, logger, model, ema_model=None, **kwargs):
        """
        Unified checkpoint loading interface, integrating all improvements

        Args:
            resume_from: checkpoint path
            logger: logger
            model: main model
            ema_model: EMA model
            **kwargs: other parameters

        Returns:
            tuple: (model, ema_model)
        """
        logger.info(f"Using unified checkpoint loading from {resume_from}")
        
        # Validate expected components
        expected_components = []
        if hasattr(model, 'vae_model') or kwargs.get('external_vae_model') is not None:
            expected_components.append('vae_model')
        if hasattr(model, 'vit_model'):
            expected_components.append('vit_model') 
        if hasattr(model, 'language_model'):
            expected_components.append('language_model')
        
        # Pre-validate checkpoint
        if resume_from and os.path.exists(resume_from):
            integrity_ok = validate_checkpoint_integrity(resume_from, expected_components, logger)
            if not integrity_ok:
                logger.warning("Checkpoint integrity pre-check failed, will try maximum compatibility loading")
        
        # Call unified loading function to load main model
        model, ema_model = FSDPCheckpoint.try_load_ckpt(resume_from, logger, model, ema_model, **kwargs)

        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config, resume_model_optimizer = True ):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            if resume_model_optimizer and os.path.exists(os.path.join(resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt")):
                optimizer_state_dict_path = os.path.join(
                    resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
                )
                optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
                optimizer.load_state_dict(optimizer_state_dict)
                del optimizer_state_dict

                scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
                scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
                scheduler.load_state_dict(scheduler_state_dict)
                del scheduler_state_dict

            # Extract step number from checkpoint directory name
            checkpoint_name = os.path.basename(os.path.normpath(resume_from))
            try:
                # Try to convert directory name to integer (for regular training checkpoints)
                train_steps = int(checkpoint_name) + 1
            except ValueError:
                # If not a number, it's a pre-trained model or specially named checkpoint
                # Check if train_state.json file exists
                train_state_path = os.path.join(resume_from, "train_state.json")
                if os.path.exists(train_state_path):
                    import json
                    with open(train_state_path, 'r') as f:
                        train_state = json.load(f)
                        train_steps = train_state.get('train_steps', 0)
                    _logger.info(f"Loaded training steps from train_state.json file: {train_steps}")
                else:
                    # If no train_state.json, default to start from 0
                    train_steps = 0
                    _logger.warning(f"checkpoint '{checkpoint_name}' is not in numeric format and train_state.json not found, will start training from step 0")
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer, 
        SiglipEncoderLayer, 
        MLPconnector, 
        Qwen2MoEDecoderLayer, 
        Qwen2MoTDecoderLayer,
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


def fsdp_ema_setup_with_ddp(ema_model, fsdp_config, ignored_modules=[]):
    """
    EMA model FSDP wrapping strategy:
    - VAE as ignored_modules, each device maintains complete copy (not sharded by FSDP)
    - Other modules normal FSDP sharding
    - VAE parameter updates through manual sync from main model DDP VAE
    """
    # All EMA model parameters don't need gradients
    for param in ema_model.parameters():
        param.requires_grad = False
    
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    
    # If EMA model has VAE, ensure it's on correct device and add to ignored_modules
    if hasattr(ema_model, 'vae_model') and ema_model.vae_model is not None:
        # Ensure EMA model's VAE is on correct GPU
        try:
            ema_model.vae_model = ema_model.vae_model.cuda(device_id)
            _logger.info(f"[Rank {rank}] EMA model's VAE moved to GPU {device_id}")
        except Exception as e:
            _logger.warning(f"[Rank {rank}] EMA model VAE move to GPU failed: {e}")
        
        ignored_modules = ignored_modules.copy()  # Avoid modifying original list
        ignored_modules.append(ema_model.vae_model)
        _logger.info(f"[Rank {rank}] EMA model's VAE added to ignored_modules (each device maintains complete copy)")
    
    # Use standard FSDP wrapping, VAE as ignored_modules won't be sharded
    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    _logger.info(f"[Rank {rank}] EMA model FSDP wrapping completed (VAE maintains complete copy on each device)")
    
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    # Handle FSDP parameters
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    # Print debug information
    total_fsdp_params = sum(p.numel() for p in ema_params) if ema_params else 0
    _logger.debug(f"Total FSDP parameters: {total_fsdp_params}")
    
    # Batch update FSDP parameters
    if ema_params:
        torch._foreach_mul_(ema_params, decay)
        torch._foreach_add_(ema_params, new_params, alpha=1 - decay)

    # Additionally handle DDP-wrapped VAE parameters (if exists)
    _update_ddp_vae_params(ema_model, model, decay)


def _update_ddp_vae_params(ema_model, model, decay):
    """
    Manually update VAE parameters: sync from main model DDP VAE to EMA model local VAE
    
    Scenario:
    - Main model: VAE wrapped with DDP (needs training)
    - EMA model: VAE as ignored_modules (each device maintains complete copy, no gradients needed)
    """
    _logger.debug("Checking VAE parameter update...")
    
    # Check if main model has DDP-wrapped VAE
    main_has_ddp_vae = (hasattr(model, 'vae_model') and model.vae_model is not None and 
                        hasattr(model.vae_model, 'module'))
    
    # Check if EMA model has local VAE (ignored_modules)
    ema_has_local_vae = (hasattr(ema_model, 'vae_model') and ema_model.vae_model is not None and 
                         not hasattr(ema_model.vae_model, 'module'))
    
    _logger.debug(f"Main model DDP VAE: {main_has_ddp_vae}, EMA model local VAE: {ema_has_local_vae}")
    
    if main_has_ddp_vae and ema_has_local_vae:
        # This is our target scenario: main model DDP VAE → EMA model local VAE
        _logger.debug("Executing DDP VAE → local VAE parameter sync")
        
        # Get parameters from main model DDP VAE (need to access through .module)
        main_vae_params = list(model.vae_model.module.parameters())
        
        # EMA model VAE parameters (direct access, no need for .module)
        ema_vae_params = list(ema_model.vae_model.parameters())
        
        # Verify parameter count consistency
        if len(main_vae_params) != len(ema_vae_params):
            _logger.error(f"VAE parameter count mismatch: main model {len(main_vae_params)}, EMA model {len(ema_vae_params)}")
            return
        
        # Calculate total parameter count
        total_vae_params = sum(p.numel() for p in ema_vae_params)
        _logger.debug(f"Total VAE parameters: {total_vae_params}")
        
        # Update EMA VAE parameters one by one
        updated_count = 0
        for ema_p, main_p in zip(ema_vae_params, main_vae_params):
            if main_p.requires_grad:  # Only update parameters that require gradients in main model
                # EMA update formula: ema_param = decay * ema_param + (1 - decay) * main_param
                ema_p.data.mul_(decay).add_(main_p.data, alpha=1 - decay)
                updated_count += 1
        
        _logger.debug(f"Successfully updated {updated_count} VAE parameters")
        
    elif not main_has_ddp_vae and not ema_has_local_vae:
        _logger.debug("Both models have no VAE, skipping VAE parameter update")
        
    else:
        _logger.debug("VAE configuration mismatch, skipping VAE parameter update")
        _logger.debug("Expected: main model DDP VAE + EMA model local VAE")
