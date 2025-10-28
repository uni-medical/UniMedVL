# Copyright (c) UniMedVL Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#%% Cell 1: Environment setup and imports
import sys
import os

TARGET_GPU_DEVICE = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = TARGET_GPU_DEVICE
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import warnings
warnings.filterwarnings('ignore')

import torch
import gc
from typing import Optional
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights, dispatch_model
from safetensors.torch import load_file, save_file

# TODO: Set this to your UniMedVL installation directory
ROOT = "/path/to/UniMedVL"
sys.path.append(ROOT)

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens, pil_img2rgb
from modeling.unimedvl import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

print("Environment initialized successfully")

#%% Cell 2: Configuration and ImageGenerator class definition
DEFAULT_CONFIG = {
    # TODO: Download the UniMedVL model checkpoint and set the path here
    "model_path": "/path/to/unimedvl_checkpoint",
    "target_gpu_device": "0",
    "max_mem_per_gpu": "40GiB",
    "enable_cpu_loading": True,
    'use_model_checkpoint': False,
    "enable_auto_bf16_conversion": True,
    "offload_folder": "/tmp/bagel_offload",
    "seed": 42,
    "vae_transform_size": (1024, 32, 16),
    "vit_transform_size": (980, 387, 14),
    # Text generation sampling parameters
    "text_do_sample": False,  # False = greedy (deterministic), True = sampling (with temperature)
    "text_temperature": 0.3,  # Only effective when text_do_sample=True. Lower = more focused, Higher = more random
}


class ImageGenerator:
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.model = None
        self.vae_model = None
        self.tokenizer = None
        self.vae_transform = None
        self.vit_transform = None
        self.new_token_ids = None
        self.inferencer = None
        self.loaded = False

    def set_seed(self, seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def convert_checkpoint_to_bf16(self, input_path, output_path):
        if not os.path.exists(input_path):
            return False

        state_dict = load_file(input_path, device="cpu")
        first_key = next(iter(state_dict))

        # Already bf16
        if state_dict[first_key].dtype == torch.bfloat16:
            if input_path != output_path:
                shutil.copy(input_path, output_path)
            del state_dict
            return True

        bf16_state_dict = {key: tensor.to(torch.bfloat16) for key, tensor in state_dict.items()}
        del state_dict
        gc.collect()

        save_file(bf16_state_dict, output_path)
        del bf16_state_dict
        gc.collect()
        return True

    def create_cpu_device_map(self, model):
        cpu_device_map = {}
        for name, _ in model.named_parameters():
            cpu_device_map[name] = "cpu"

        cpu_device_map.update({
            'language_model': 'cpu', 'vit_model': 'cpu', 'time_embedder': 'cpu',
            'latent_pos_embed': 'cpu', 'vae2llm': 'cpu', 'llm2vae': 'cpu',
            'connector': 'cpu', 'vit_pos_embed': 'cpu', 'vae_model': 'cpu'
        })
        return cpu_device_map

    def load_weights_progressively(self, model, vae_model, model_path):
        cpu_device_map = self.create_cpu_device_map(model)

        if not model_path:
            raise ValueError("model_path required")

        # Determine checkpoint filename
        original_checkpoint_name = "model.safetensors" if self.config['use_model_checkpoint'] else "ema.safetensors"
        bf16_checkpoint_name = "model_bf16.safetensors" if self.config['use_model_checkpoint'] else "ema_bf16.safetensors"

        bf16_checkpoint_path = os.path.join(model_path, bf16_checkpoint_name)
        original_checkpoint_path = os.path.join(model_path, original_checkpoint_name)

        final_checkpoint_path = None

        # Try bf16 first
        if os.path.exists(bf16_checkpoint_path):
            final_checkpoint_path = bf16_checkpoint_path
        elif os.path.exists(original_checkpoint_path) and self.config['enable_auto_bf16_conversion']:
            success = self.convert_checkpoint_to_bf16(original_checkpoint_path, bf16_checkpoint_path)
            final_checkpoint_path = bf16_checkpoint_path if success else original_checkpoint_path
        elif os.path.exists(original_checkpoint_path):
            final_checkpoint_path = original_checkpoint_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        model = load_checkpoint_and_dispatch(
            model, checkpoint=final_checkpoint_path, device_map=cpu_device_map,
            offload_buffers=False, dtype=torch.bfloat16, force_hooks=False
        )

        torch.cuda.empty_cache()
        gc.collect()

        return model, vae_model

    def deploy_to_gpu_unified(self, model, vae_model, target_device="cuda:0"):
        if not torch.cuda.is_available():
            return model, vae_model

        if torch.cuda.device_count() == 1:
            device_map = infer_auto_device_map(
                model, max_memory={0: self.config['max_mem_per_gpu']},
                no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
            )

            # Ensure related modules on same device
            same_device_modules = [
                'language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed',
                'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed'
            ]

            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for module_name in same_device_modules:
                if module_name in device_map:
                    device_map[module_name] = first_device
        else:
            device_map = {name: target_device for name, _ in model.named_parameters()}
            first_device = target_device

        model = dispatch_model(model, device_map=device_map)
        vae_model = vae_model.to(device=first_device, dtype=torch.bfloat16)

        return model, vae_model

    def load_model(self):
        if self.loaded:
            print("Model already loaded")
            return

        print("Loading Bagel model (may take 5-10 minutes)...")
        self.set_seed(self.config['seed'])

        model_path = self.config.get('model_path')
        if not model_path:
            raise ValueError("model_path required")

        print(f"Checkpoint path: {model_path}")

        # Load configs
        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        # Load VAE
        vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
        vae_model = vae_model.cpu().to(torch.bfloat16)

        # Bagel config
        config = BagelConfig(
            visual_gen=True, visual_und=True,
            llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
            vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh',
            latent_patch_size=2, max_latent_size=64,
        )

        # Initialize empty model
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config, vae_model=vae_model)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids

        # Image transforms
        vae_size = self.config['vae_transform_size']
        vit_size = self.config['vit_transform_size']
        self.vae_transform = ImageTransform(vae_size[0], vae_size[1], vae_size[2])
        self.vit_transform = ImageTransform(vit_size[0], vit_size[1], vit_size[2])

        # Load weights
        if self.config['enable_cpu_loading']:
            model, vae_model = self.load_weights_progressively(model, vae_model, model_path)
            torch.cuda.empty_cache()
            gc.collect()

        # Deploy to GPU
        target_device = f"cuda:{self.config['target_gpu_device']}" if torch.cuda.is_available() else "cpu"
        model, vae_model = self.deploy_to_gpu_unified(model, vae_model, target_device)

        model = model.eval()
        self.model = model
        self.vae_model = vae_model

        # Create inferencer
        self.inferencer = InterleaveInferencer(
            model=self.model, vae_model=self.vae_model, tokenizer=self.tokenizer,
            vae_transform=self.vae_transform, vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids
        )

        self.loaded = True
        print("Model loaded successfully")
        self.show_gpu_memory()

    def show_gpu_memory(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {allocated:.1f}GB / {total:.1f}GB ({allocated/total*100:.1f}%)")

print("ImageGenerator class defined")

#%% Cell 3: Initialize and load model
generator = ImageGenerator(DEFAULT_CONFIG)
generator.load_model()

#%% Cell 4: Image editing example
# Modify these parameters for your task
# TODO: Replace with your input image path
image_path = "/path/to/your/test_image.png"
edit_instruction = """
Synthesize a HER2 IHC image with 2+ expression level from the given H&E stained pathology input, maintaining all anatomical structures and generating realistic 2+ immunohistochemistry patterns.
"""

# Inference mode: use_thinking enables deep reasoning (slower but more thoughtful)
# Default: use image understanding mode for context-aware editing
use_thinking = False

# Generation parameters
cfg_text_scale = 4.0
cfg_img_scale = 2.0
num_timesteps = 50
timestep_shift = 3.0
seed = 42

print("Starting image editing...")

if generator.loaded and os.path.exists(image_path):
    input_image = Image.open(image_path)

    # Convert to RGB
    if input_image.mode != 'RGB':
        if input_image.mode == 'RGBA':
            background = Image.new('RGB', input_image.size, (255, 255, 255))
            background.paste(input_image, mask=input_image.split()[-1])
            input_image = background
        else:
            input_image = input_image.convert('RGB')

    if seed > 0:
        generator.set_seed(seed)

    final_instruction = edit_instruction

    # Mode selection: thinking mode vs understanding mode
    if use_thinking:
        # Thinking mode: deep reasoning with internal thought process
        analysis_prompt = f"Analyze this medical image and develop an editing strategy for: '{edit_instruction}'."
        thinking_result = generator.inferencer(
            image=input_image, text=analysis_prompt, think=True,
            understanding_output=True,
            do_sample=generator.config['text_do_sample'],
            text_temperature=generator.config['text_temperature'],
            max_think_token_n=800
        )
        analysis_text = thinking_result.get('text', '')
        if analysis_text:
            print(f"Thinking mode - Analysis:\n{analysis_text}")
            final_instruction = f"{edit_instruction}\n\nBased on analysis: {analysis_text}"
    else:
        # Understanding mode (default): generate contextual description before editing
        understanding_result = generator.inferencer(
            image=input_image, text=edit_instruction, think=False,
            understanding_output=True,
            do_sample=generator.config['text_do_sample'],
            text_temperature=generator.config['text_temperature'],
            max_think_token_n=800
        )
        understanding_text = understanding_result.get('text', '')
        if understanding_text:
            print(f"Understanding mode - Context:\n{understanding_text}")
            final_instruction = f"{edit_instruction}\n\n{understanding_text}"

    # Image editing
    original_width, original_height = input_image.size
    target_size = generator.inferencer._calculate_target_size_with_aspect_ratio(
        original_width, original_height
    )

    input_list = [input_image, final_instruction]

    edit_result = generator.inferencer.interleave_inference(
        input_lists=input_list, think=False, understanding_output=False,
        cfg_text_scale=cfg_text_scale, cfg_img_scale=cfg_img_scale,
        cfg_interval=[0.0, 1.0], cfg_renorm_type="text_channel",
        timestep_shift=timestep_shift, num_timesteps=num_timesteps,
        image_shapes=target_size
    )

    edited_image = None
    for item in edit_result:
        if isinstance(item, Image.Image):
            edited_image = item
            break

    if edited_image:
        if edited_image.mode != 'RGB':
            edited_image = edited_image.convert('RGB')

        # Display comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        ax1.imshow(input_image)
        ax1.set_title("Original")
        ax1.axis('off')
        ax2.imshow(edited_image)
        ax2.set_title("Edited")
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
        print("Image editing completed successfully")
    else:
        print("Image editing failed")
else:
    print("Model not loaded or image file not found")

#%%
