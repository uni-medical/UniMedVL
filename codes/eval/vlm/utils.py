# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# Copyright (c) UniMedVL Project
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
# This file has been further modified by UniMedVL  team.
#
# UniMedVL modifications:
# - Enhanced model loading functionality with CPU-optimized and GPU-direct loading modes
# - Added support for progressive weight loading and memory-efficient checkpoint handling
# - Improved model deployment strategies for multi-GPU and single-GPU configurations
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

import os
import yaml

from data.data_utils import add_special_tokens, pil_img2rgb
from modeling.unimedvl import (
    BagelConfig, 
    Bagel, 
    Qwen2Config, 
    Qwen2ForCausalLM, 
    SiglipVisionConfig, 
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from safetensors.torch import load_file

from data.transforms import ImageTransform


def load_model_and_tokenizer(args):
    """Load Bagel model and tokenizer for VQA/understanding tasks (no image generation)."""
    use_cpu_optimization = getattr(args, 'cpu_optimized_loading', False) and getattr(args, 'use_fast_chat_api', False)

    if use_cpu_optimization:
        return _load_model_cpu_optimized(args)
    else:
        return _load_model_original(args)


def _load_model_original(args):
    """Original model loading - direct GPU loading for training compatibility."""
    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module ="Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    config = BagelConfig(
        visual_gen=False,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
    )
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    if hasattr(args, "checkpoint_weight_path") and args.checkpoint_weight_path is not None:
        # First, try to find the bfloat16 checkpoint, then fall back to the standard one.
        finetuned_path_bf16 = os.path.join(args.checkpoint_weight_path, "ema_bf16.safetensors")
        finetuned_path_f32 = os.path.join(args.checkpoint_weight_path, "ema.safetensors")
        
        if os.path.exists(finetuned_path_bf16):
            model_state_dict_path = finetuned_path_bf16
        elif os.path.exists(finetuned_path_f32):
            model_state_dict_path = finetuned_path_f32
        else:
            raise FileNotFoundError(
                f"Could not find 'ema_bf16.safetensors' or 'ema.safetensors' in {args.checkpoint_weight_path}"
            )

        print(f"-------------------->>>> Loading model from: {model_state_dict_path}. First loading old model from {os.path.join(args.model_path, 'ema.safetensors')} to ensure no missing weights.")

        old_model_state_dict_path = os.path.join(args.model_path, "ema.safetensors")
        old_model_state_dict = load_file(old_model_state_dict_path, device="cpu")
        msg = model.load_state_dict(old_model_state_dict, strict=False)
        print(msg)
        del old_model_state_dict

        
        model_state_dict = load_file(model_state_dict_path, device="cpu")
        msg = model.load_state_dict(model_state_dict, strict=False)
        print(msg)
        del model_state_dict
        model = model.cuda().eval()
        
    else:
        model_state_dict_path = os.path.join(args.model_path, "ema.safetensors")
        print(f"-------------------->>>> Loading model from: {model_state_dict_path}")

        model_state_dict = load_file(model_state_dict_path, device="cpu")
        msg = model.load_state_dict(model_state_dict, strict=False)
        print(msg)
        del model_state_dict
        model = model.cuda().eval()

    return model, tokenizer, new_token_ids


def _load_model_cpu_optimized(args):
    """CPU-optimized model loading - memory-friendly for inference."""
    import os
    import gc
    import torch

    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    config = BagelConfig(
        visual_gen=False,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
    )

    original_device = torch.get_default_device() if hasattr(torch, 'get_default_device') else None
    torch.set_default_device('cpu')
    try:
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

        language_model = language_model.cpu()
        vit_model = vit_model.cpu()
        model = model.cpu()

        assert all(p.device.type == 'cpu' for p in model.parameters()), "Some params not on CPU"
    finally:
        if original_device is not None:
            torch.set_default_device(original_device)
        else:
            try:
                torch.set_default_device(None)
            except Exception:
                pass

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    base_ckpt_path = os.path.join(args.model_path, "ema.safetensors")
    if not os.path.exists(base_ckpt_path):
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt_path}")
    print(f"-------------------->>>> You are loading the model from: {base_ckpt_path}")
    base_state = load_file(base_ckpt_path, device="cpu")
    msg = model.load_state_dict(base_state, strict=False)
    print(msg)
    del base_state

    if hasattr(args, "checkpoint_weight_path") and args.checkpoint_weight_path:
        ft_bf16 = os.path.join(args.checkpoint_weight_path, "ema_bf16.safetensors")
        ft_f32 = os.path.join(args.checkpoint_weight_path, "ema.safetensors")

        if os.path.exists(ft_bf16):
            ft_path = ft_bf16
        elif os.path.exists(ft_f32):
            ft_path = ft_f32
        else:
            raise FileNotFoundError(
                f"Could not find 'ema_bf16.safetensors' or 'ema.safetensors' in {args.checkpoint_weight_path}"
            )

        print(f"-------------------->>>> Loading from: {ft_path}. First loaded old model from {base_ckpt_path}.")
        ft_state = load_file(ft_path, device="cpu")
        msg = model.load_state_dict(ft_state, strict=False)
        print(msg)
        del ft_state

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            model = model.to(dtype=torch.bfloat16)
            model = model.to(device='cuda', dtype=torch.bfloat16).eval()
        except RuntimeError as e:
            print(f"GPU move failed: {e}")
            raise
    else:
        model = model.eval()

    return model, tokenizer, new_token_ids


def _load_model_cpu_optimized1(args):
    """
    Simplified CPU-optimized model loading aligned with original logic:
    1. Create model structure on CPU
    2. Load base weights on CPU (args.model_path/ema.safetensors)
    3. If fine-tuned weights provided, prioritize bf16 ema_bf16.safetensors, otherwise fallback to ema.safetensors
    4. Merge weights on CPU (fine-tuned weights override base weights)
    5. Save merged weights to temp file, dispatch to GPU with accelerate's load_checkpoint_and_dispatch
    """
    import os
    import gc
    import torch
    import tempfile
    from safetensors.torch import load_file, save_file
    from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch

    print("--- Loading Model using CPU-First Approach (with base+finetune merge) ---")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    original_device = torch.get_default_device() if hasattr(torch, 'get_default_device') else None
    torch.set_default_device('cpu')

    tmp_merged_path = None
    offload_folder = None
    try:
        llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        config = BagelConfig(
            visual_gen=False,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
        )

        print("--- Creating model on CPU ---")
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

        tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            max_memory = {0: f"{int(total_memory * 0.98 / 1024**3)}GiB"}
            print(f"---------------->>>> GPU Memory limit set to: {max_memory[0]}")
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[
                    "Qwen2MoTDecoderLayer",  # 与 llm_config.layer_module 对齐
                    "Qwen2DecoderLayer",     # 兼容一些实现命名
                    "SiglipEncoderLayer",
                ]
            )
            print(f"Device map: {device_map}")
        else:
            device_map = "cpu"

        base_checkpoint_path = os.path.join(args.model_path, "ema.safetensors")
        if not os.path.exists(base_checkpoint_path):
            raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint_path}")
        print(f"Base checkpoint: {base_checkpoint_path}")

        final_checkpoint_path = base_checkpoint_path
        if hasattr(args, "checkpoint_weight_path") and args.checkpoint_weight_path:
            finetuned_path_bf16 = os.path.join(args.checkpoint_weight_path, "ema_bf16.safetensors")
            finetuned_path_f32 = os.path.join(args.checkpoint_weight_path, "ema.safetensors")

            if os.path.exists(finetuned_path_bf16):
                ft_path = finetuned_path_bf16
                print(f"Found fine-tuned bf16 checkpoint: {ft_path}")
            elif os.path.exists(finetuned_path_f32):
                ft_path = finetuned_path_f32
                print(f"Found fine-tuned float32 checkpoint: {ft_path}")
            else:
                raise FileNotFoundError(
                    f"Could not find 'ema_bf16.safetensors' or 'ema.safetensors' in {args.checkpoint_weight_path}"
                )

            print("--- Loading and merging checkpoints on CPU ---")
            base_state = load_file(base_checkpoint_path, device="cpu")
            ft_state = load_file(ft_path, device="cpu")
            base_state.update(ft_state)
            del ft_state
            gc.collect()

            offload_folder = tempfile.mkdtemp(prefix="bagel_offload_")
            fd, tmp_merged_path = tempfile.mkstemp(prefix="bagel_merged_", suffix=".safetensors", dir=offload_folder)
            os.close(fd)
            save_file(base_state, tmp_merged_path)
            del base_state
            gc.collect()

            final_checkpoint_path = tmp_merged_path
            print(f"Merged checkpoint saved to: {final_checkpoint_path}")
        else:
            print("--- No fine-tuned checkpoint provided; using base checkpoint only ---")

        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=final_checkpoint_path,
            device_map=device_map,
            offload_buffers=False,
            offload_state_dict=False,
            offload_folder=offload_folder,
            dtype=torch.bfloat16,
            force_hooks=False,
        )

        print("✅ Model loaded and dispatched")
        model = model.eval()
        return model, tokenizer, new_token_ids

    finally:
        try:
            if tmp_merged_path and os.path.exists(tmp_merged_path):
                os.remove(tmp_merged_path)
        except Exception:
            pass
        if original_device is not None:
            torch.set_default_device(original_device)
        else:
            torch.set_default_device(None)


def _load_model_cpu_optimized0(args):
    """
    CPU-optimized model loading method for memory-efficient inference
    """
    import torch
    import gc
    from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
    
    print("--- Loading Model using Fast Chat API (CPU-Optimized) ---")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    config = BagelConfig(
        visual_gen=False,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
    )

    print("--- Initializing empty model structure ---")
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    max_mem_per_gpu = getattr(args, 'max_mem_per_gpu', '40GiB')

    print("--- Preparing device map for GPU deployment ---")
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    print("Device Map:", device_map)

    same_device_modules = [
        'language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed',
        'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed'
    ]
    if torch.cuda.device_count() > 0:
        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
        else:
            first_device = device_map.get(same_device_modules[0])
        
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    print("--- Loading base checkpoint on CPU ---")
    base_checkpoint_path = os.path.join(args.model_path, "ema.safetensors")
    base_state_dict = load_file(base_checkpoint_path, device="cpu")
    print(f"Base checkpoint loaded from: {base_checkpoint_path}")

    final_state_dict = base_state_dict
    if hasattr(args, "checkpoint_weight_path") and args.checkpoint_weight_path is not None:
        print("--- Loading and merging fine-tuned checkpoint on CPU ---")

        finetuned_path_bf16 = os.path.join(args.checkpoint_weight_path, "ema_bf16.safetensors")
        finetuned_path_f32 = os.path.join(args.checkpoint_weight_path, "ema.safetensors")
        
        if os.path.exists(finetuned_path_bf16):
            model_state_dict_path = finetuned_path_bf16
            print(f"Found bfloat16 checkpoint: {model_state_dict_path}")
        elif os.path.exists(finetuned_path_f32):
            model_state_dict_path = finetuned_path_f32
            print(f"Found float32 checkpoint: {model_state_dict_path}")
        else:
            raise FileNotFoundError(
                f"Could not find 'ema_bf16.safetensors' or 'ema.safetensors' in {args.checkpoint_weight_path}"
            )

        finetuned_state_dict = load_file(model_state_dict_path, device="cpu")

        print("--- Merging checkpoints on CPU ---")
        final_state_dict.update(finetuned_state_dict)

        del finetuned_state_dict
        gc.collect()
        print("Checkpoints merged successfully on CPU")

    print("--- Dispatching merged model to GPU (SINGLE dispatch) ---")

    import tempfile
    temp_checkpoint_path = "/tmp/merged_checkpoint.safetensors"
    
    print(f"Saving merged checkpoint to temporary file: {temp_checkpoint_path}")
    from safetensors.torch import save_file
    save_file(final_state_dict, temp_checkpoint_path)

    del final_state_dict, base_state_dict
    torch.cuda.empty_cache()
    gc.collect()

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=temp_checkpoint_path,
        device_map=device_map,
        offload_buffers=False,
        offload_state_dict=False,
        dtype=torch.bfloat16,
        force_hooks=False
    )

    if os.path.exists(temp_checkpoint_path):
        os.remove(temp_checkpoint_path)
        print(f"Temporary checkpoint file removed: {temp_checkpoint_path}")

    model = model.eval()

    if torch.cuda.is_available():
        print("--- Final GPU Memory Status (Fast Chat API CPU-Optimized) ---")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i} - Final memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    print('Fast Chat API model loaded successfully with CPU-optimized approach (single GPU dispatch).')
    return model, tokenizer, new_token_ids


def build_transform():
    with open("/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/medical_unified_project/src/Bagel-main/data/configs/example.yaml", "r") as f:
        data_config = yaml.safe_load(f)

    max_image_size = data_config['vlm_sft']['image_transform_args']['max_image_size']
    min_image_size = data_config['vlm_sft']['image_transform_args']['min_image_size']
    image_stride = data_config['vlm_sft']['image_transform_args']['image_stride']
    max_pixels = data_config['vlm_sft']['image_transform_args']['max_pixels']

    image_transform = ImageTransform(
        max_image_size=max_image_size,
        min_image_size=min_image_size,
        image_stride=image_stride,
        max_pixels=max_pixels,
    )

    return image_transform

def process_conversation(images, conversation):
    images = [pil_img2rgb(image) for image in images]
    return images, conversation
