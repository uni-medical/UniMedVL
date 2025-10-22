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
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import gc
import random
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any
from datetime import datetime
import shutil
import time

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
from eval.vlm.utils import process_conversation, build_transform

print("Environment initialized successfully")

#%% Cell 2: Configuration and VQAInferencer class
DEFAULT_CONFIG = {
    # TODO: Download the UniMedVL model checkpoint and set the path here
    "model_path": "/path/to/unimedvl_checkpoint",
    "target_gpu_device": "0",
    "max_mem_per_gpu": "40GiB",
    "temperature": 1.0,
    "max_new_tokens": 512,
    "do_sample": True,
    "seed": 42,
    "enable_cpu_loading": True,
    "enable_auto_bf16_conversion": True,
    "use_model_checkpoint": False,  # False = ema.safetensors, True = model.safetensors
    "offload_folder": "/tmp/bagel_offload",
}


class VQAInferencer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or DEFAULT_CONFIG
        self.model = None
        self.tokenizer = None
        self.new_token_ids = None
        self.image_transform = None
        self.loaded = False

    def set_seed(self, seed):
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
            'language_model': 'cpu', 'vit_model': 'cpu',
            'connector': 'cpu', 'vit_pos_embed': 'cpu'
        })
        return cpu_device_map

    def load_weights_progressively(self, model, model_path):
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

        return model

    def deploy_to_gpu_unified(self, model, target_device="cuda:0"):
        if not torch.cuda.is_available():
            return model

        if torch.cuda.device_count() == 1:
            device_map = infer_auto_device_map(
                model, max_memory={0: self.config['max_mem_per_gpu']},
                no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
            )

            # Ensure related modules on same device
            same_device_modules = [
                'language_model.model.embed_tokens',
                'connector', 'vit_pos_embed'
            ]

            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for module_name in same_device_modules:
                if module_name in device_map:
                    device_map[module_name] = first_device
        else:
            device_map = {name: target_device for name, _ in model.named_parameters()}
            first_device = target_device

        model = dispatch_model(model, device_map=device_map)

        return model

    def load_model(self):
        if self.loaded:
            print("Model already loaded")
            return

        print("Loading model (may take 5-10 minutes)...")
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

        config = BagelConfig(
            visual_gen=False,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
        )

        # Build empty model
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids

        # Image transform
        self.image_transform = build_transform()

        # Load weights
        if self.config['enable_cpu_loading']:
            model = self.load_weights_progressively(model, model_path)
            torch.cuda.empty_cache()
            gc.collect()

        # Deploy to GPU
        target_device = f"cuda:{self.config['target_gpu_device']}" if torch.cuda.is_available() else "cpu"
        model = self.deploy_to_gpu_unified(model, target_device)

        model = model.eval()
        self.model = model

        self.loaded = True
        print("Model loaded successfully")
        self.show_gpu_memory()

    def show_gpu_memory(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {allocated:.1f}GB / {total:.1f}GB ({allocated/total*100:.1f}%)")

    def cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        self.show_gpu_memory()

    def infer_single(
        self,
        image_path,
        prompt,
        temperature=None,
        max_new_tokens=None,
        do_sample=None,
        show_image=False
    ):
        if not self.loaded:
            raise RuntimeError("Model not loaded, please call load_model() first")

        temperature = temperature if temperature is not None else self.config['temperature']
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config['max_new_tokens']
        do_sample = do_sample if do_sample is not None else self.config['do_sample']

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        input_image = Image.open(image_path).convert("RGB")
        print(f"Loaded image: {input_image.size}")

        if show_image:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 6))
                plt.imshow(input_image)
                plt.axis('off')
                plt.title('Input Image')
                plt.show()
            except:
                print("  Cannot display image")

        print(f"Executing inference...")
        start_time = time.time()

        # Fast Chat API inference
        images, conversation = process_conversation([input_image], prompt)
        
        print(f"Executing inference...1")
        
        response = self.model.chat(
            self.tokenizer,
            self.new_token_ids,
            self.image_transform,
            images=images,
            prompt=conversation,
            max_length=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

        answer = response
        inference_time = time.time() - start_time

        result = {
            'answer': answer,
            'input_image': input_image,
            'time': inference_time,
            'image_path': image_path,
            'prompt': prompt,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        print(f"Inference completed! Time: {inference_time:.2f}s")
        return result

print("VQAInferencer class defined")

#%% Cell 3: Initialize and load model
inferencer = VQAInferencer(DEFAULT_CONFIG)
inferencer.load_model()

#%% Cell 4: VQA inference
# Modify these parameters for your task
# TODO: Replace with your input image path
image_path = "/path/to/your/test_image.png"
prompt = "Please analyze this chest X-ray using professional medical knowledge: 1) How is the transparency of the lung fields? 2) Are there any nodules, masses, or infiltrative lesions present? 3) Is the pleura smooth? 4) Is the contour of the cardiac silhouette normal? 5) How is the mediastinal structure? Please provide a detailed radiological report and diagnostic opinions."

temperature = 0.1
max_new_tokens = 512
show_image = True

result = inferencer.infer_single(
    image_path=image_path,
    prompt=prompt,
    temperature=temperature,
    max_new_tokens=max_new_tokens,
    show_image=show_image
)

print(f"\nAnswer:")
print(result['answer'])
print(f"\nInference time: {result['time']:.2f}s")

#%%