# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 Unimedvl Team
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by Unimedvl Team.
# Modifications: Enhanced medical image editing dataset with improved data validation and debugging capabilities.

import json
import os
import traceback
from PIL import Image, ImageFile, PngImagePlugin

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from interleave_datasets.interleave_t2i_dataset import InterleavedBaseIterableDataset
from data_utils import pil_img2rgb
from distributed_iterable_dataset import DistributedIterableDataset

Image.MAX_IMAGE_PIXELS = 20000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class MedicalImageEditingIterableDataset_ver1(InterleavedBaseIterableDataset):
    """
    
    sequence_plan = [
        {
            'type': 'vae_image',
            'enable_cfg': 1,
            'loss': 0,
            'special_token_loss': 0,
            'special_token_label': None,
        },
        
        {
            'type': 'vit_image', 
            'enable_cfg': 0,
            'loss': 0,
            'special_token_loss': 0,
            'special_token_label': None,
        },
        
        {
            'type': 'text',
            'enable_cfg': 0,
            'loss': 0,
            'special_token_loss': 0,
            'special_token_label': None,
        },
        
        {
            'type': 'text',
            'enable_cfg': 1,
            'loss': 1,
            'special_token_loss': 0,
            'special_token_label': None,
        },
        
        {
            'type': 'vae_image',
            'enable_cfg': 0,
            'loss': 1,
            'special_token_loss': 0,
            'special_token_label': None,
        }
    ]
    
    Medical image generation dataset loader
    
    Data format:
    
    {
        "main_task_type": "image_generation",
        "sub_task_type": "medical_vqa", 
        "dataset": "CFP_train_val_clean_filter_clip_vqa",
        "input_img": [{"path": "/path/to/input.jpeg", "height": 512, "width": 512, ...}],
        "output_img": [{"path": "/path/to/output.jpeg", "height": 512, "width": 512, ...}],
        "message": [
            {"from": "human", "value": "Using the description \"...\" generate a medical image."},
            {"from": "gpt", "value": "This is the resulting image."}
        ]
    }
    
    Sequence construction strategy:
    Input Image(Condition) -> User Instruction(Condition) -> GPT Response(Target) -> Output Image(Target)
    need_loss:  False      ->   False                    ->   True             ->    True
    need_vae:   True       ->     -                      ->     -              ->   False  
    need_vit:   True       ->     -                      ->     -              ->   False
    """
    
    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform,
        jsonl_path_list, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
        shuffle_lines=False, shuffle_seed=0,
    ):
        """
        Initialize medical image generation dataset
        
        Args:
            dataset_name: Dataset name
            transform: VAE image transform
            tokenizer: Text tokenizer
            vit_transform: ViT image transform
            jsonl_path_list: JSONL file path list
            data_dir_list: Image directory list
            num_used_data: Number of data to use from each source
            Other parameters: Distributed training related parameters
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        
        self.data_paths = self.get_data_paths(
            jsonl_path_list, 
            data_dir_list, 
            num_used_data, 
            shuffle_lines, 
            shuffle_seed,
        )
        self.set_epoch()

    def get_data_paths(
        self, 
        jsonl_path_list, 
        data_dir_list, 
        num_used_data, 
        shuffle_lines, 
        shuffle_seed,
    ):
        """"""
        data_paths = []
        for jsonl_path, image_dir, num_data_point in zip(
            jsonl_path_list, data_dir_list, num_used_data
        ):
            with open(jsonl_path, 'r') as f:
                raw_data = f.readlines()
            
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            
            if num_data_point == 0:
                raw_data = raw_data
            else:
                raw_data = raw_data[:num_data_point]
            
            print(f"--->> Medical data from {jsonl_path}: {len(raw_data)}")
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])
        
        return data_paths

    def _validate_data_item(self, data_item):
        """"""
        if not data_item.get("input_img"):
            return False, "Missing input_img"
        if not data_item.get("output_img"):
            return False, "Missing output_img"
        if not data_item.get("message"):
            return False, "Missing message"
        
        if not isinstance(data_item["input_img"], list) or len(data_item["input_img"]) == 0:
            return False, "Invalid input_img format"
        if not isinstance(data_item["output_img"], list) or len(data_item["output_img"]) == 0:
            return False, "Invalid output_img format"
        
        if not isinstance(data_item["message"], list) or len(data_item["message"]) < 2:
            return False, "Invalid message format"
        
        return True, ""

    def _extract_messages(self, message_list):
        """messagehumangpt"""
        human_message = ""
        gpt_message = ""
        
        for msg in message_list:
            if msg.get("from") == "human":
                human_message = msg.get("value", "")
            elif msg.get("from") == "gpt":
                gpt_message = msg.get("value", "")
        
        return human_message, gpt_message

    def _load_image_safely(self, image_path, row_idx):
        """"""
        if not os.path.exists(image_path):
            print(f'Image file not found: {image_path} in row#{row_idx}')
            return None
            
        if not os.path.isfile(image_path):
            print(f'Path is not a file: {image_path} in row#{row_idx}')
            return None
            
        try:
            with Image.open(image_path) as img:
                img.verify()
                
            with Image.open(image_path) as img:
                max_pixels = getattr(Image, "MAX_IMAGE_PIXELS", 89478485)
                if img.width * img.height > max_pixels:
                    print(f"Skipping large image (>{max_pixels / 1e6:.1f}M pixels): {image_path} with size {img.size} in row#{row_idx}")
                    return None
                if img.width >= 4096 or img.height >= 4096:
                    print(f"Skipping large image (>4096x4096): {image_path} with size {img.size} in row#{row_idx}")
                    return None
                
                return pil_img2rgb(img.copy())
        except Exception as e:
            print(f"Error loading image {image_path}: {e} in row#{row_idx}")
            return None

    def parse_row(self, json_line, image_dir, row_idx):
        """
        Parse single row data and build sequence
        
        Sequence construction logic:
        1. Input image as condition (need_loss=False, need_vae=True, need_vit=True)
        2. User instruction as condition (need_loss=False)
        3. GPT response as text target (need_loss=True)  
        4. Output image as final target (need_loss=True, need_vae=False, need_vit=False)
        """
        try:
            data_item = json.loads(json_line)
            
            is_valid, error_msg = self._validate_data_item(data_item)
            if not is_valid:
                print(f"Skipping invalid data in row#{row_idx}: {error_msg}")
                return {}
            
            data = self._init_data()
            
            input_img_info = data_item["input_img"][0]
            input_img_rel_path = input_img_info["path"].lstrip(os.sep)
            input_img_path = os.path.join(image_dir, input_img_rel_path)
            
            input_image = self._load_image_safely(input_img_path, row_idx)
            if input_image is None:
                print(f"Failed to load input image: {input_img_path} in row#{row_idx}")
                return {}
            
            data = self._add_image(
                data, 
                input_image,
                need_loss=False,
                need_vae=True,
                need_vit=True,
                enable_cfg=0     
            )
            
            human_instruction, gpt_response = self._extract_messages(data_item["message"])
            
            if not human_instruction.strip():
                print(f"Empty human instruction in row#{row_idx}")
                return {}
            
            data = self._add_text(data, human_instruction.strip(), need_loss=False, enable_cfg=0)
            
            if gpt_response.strip():
                data = self._add_text(data, gpt_response.strip(), need_loss=True, enable_cfg=0)
            
            output_img_info = data_item["output_img"][0]
            output_img_rel_path = output_img_info["path"].lstrip(os.sep)
            output_img_path = os.path.join(image_dir, output_img_rel_path)
            
            output_image = self._load_image_safely(output_img_path, row_idx)
            if output_image is None:
                print(f"Failed to load output image: {output_img_path} in row#{row_idx}")
                return {}
            
            data = self._add_image(
                data,
                output_image,
                need_loss=True,
                need_vae=False,
                need_vit=False,
                enable_cfg=0
            )
            
            
            
            return data
            
        except Exception as e:
            print(f"Error processing row#{row_idx}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return {}

    def inspect_data_structure(self, data, detailed=True, show_tensor_shapes=True):
        """
        Inspect and print detailed data structure information
        
        Args:
            data: Data dictionary returned from parse_row
            detailed: Whether to show detailed information
            show_tensor_shapes: Whether to show tensor shape information
        """
        if not data:
            print("❌ Data is empty!")
            return
            
        print("\n" + "="*80)
        print("🔍 MEDICAL IMAGE GENERATION DATA STRUCTURE INSPECTION")
        print("="*80)
        
        print(f"📊 Basic Info:")
        print(f"   Total tokens: {data.get('num_tokens', 0)}")
        print(f"   Sequence length: {len(data.get('sequence_plan', []))}")
        print(f"   Text segments: {len(data.get('text_ids_list', []))}")
        print(f"   Image tensors: {len(data.get('image_tensor_list', []))}")
        
        sequence_plan = data.get('sequence_plan', [])
        if sequence_plan:
            print(f"\n📋 Sequence Plan Analysis:")
            print(f"   {'Step':<4} {'Type':<12} {'Loss':<4} {'CFG':<3} {'VAE':<3} {'ViT':<3} {'Description'}")
            print(f"   {'-'*4} {'-'*12} {'-'*4} {'-'*3} {'-'*3} {'-'*3} {'-'*20}")
            
            vae_count = 0
            vit_count = 0
            text_count = 0
            
            for i, step in enumerate(sequence_plan):
                step_type = step.get('type', 'unknown')
                loss = step.get('loss', 0)
                cfg = step.get('enable_cfg', 0)
                
                if step_type == 'vae_image':
                    vae_count += 1
                    if loss == 0 and cfg == 1:
                        desc = f"Input Image #{vae_count} (Condition)"
                    elif loss == 1 and cfg == 0:
                        desc = f"Output Image #{vae_count} (Target)"
                    else:
                        desc = f"VAE Image #{vae_count}"
                elif step_type == 'vit_image':
                    vit_count += 1
                    desc = f"ViT Encoding #{vit_count} (Vision)"
                elif step_type == 'text':
                    text_count += 1
                    if loss == 0:
                        desc = f"Text #{text_count} (Instruction)"
                    else:
                        desc = f"Text #{text_count} (Response)"
                else:
                    desc = "Unknown"
                
                vae_mark = "✓" if step_type == 'vae_image' else "-"
                vit_mark = "✓" if step_type == 'vit_image' else "-"
                
                print(f"   {i+1:<4} {step_type:<12} {loss:<4} {cfg:<3} {vae_mark:<3} {vit_mark:<3} {desc}")
        
        text_ids_list = data.get('text_ids_list', [])
        if text_ids_list and detailed:
            print(f"\n📝 Text Content Analysis:")
            for i, text_ids in enumerate(text_ids_list):
                text_length = len(text_ids)
                try:
                    if hasattr(self, 'tokenizer') and self.tokenizer:
                        decoded_text = self.tokenizer.decode(text_ids)
                        if len(decoded_text) > 100:
                            preview = decoded_text[:97] + "..."
                        else:
                            preview = decoded_text
                        print(f"   Text #{i+1}: {text_length} tokens")
                        print(f"   Content: \"{preview}\"")
                    else:
                        print(f"   Text #{i+1}: {text_length} tokens (tokenizer not available)")
                except:
                    print(f"   Text #{i+1}: {text_length} tokens (decode failed)")
                print()
        
        image_tensor_list = data.get('image_tensor_list', [])
        if image_tensor_list and show_tensor_shapes:
            print(f"🖼️  Image Tensor Analysis:")
            for i, tensor in enumerate(image_tensor_list):
                if hasattr(tensor, 'shape'):
                    shape = tensor.shape
                    if len(shape) == 3:  # [C, H, W]
                        c, h, w = shape
                        pixels = h * w
                        print(f"   Tensor #{i+1}: {shape} ({c} channels, {h}x{w} = {pixels:,} pixels)")
                        
                        if hasattr(self, 'transform') and hasattr(self.transform, 'stride'):
                            expected_tokens = pixels // (self.transform.stride ** 2)
                            print(f"   Expected tokens: {expected_tokens}")
                    else:
                        print(f"   Tensor #{i+1}: {shape}")
                else:
                    print(f"   Tensor #{i+1}: {type(tensor)} (no shape info)")
        
        print(f"\n🔄 Sequence Flow Visualization:")
        flow_parts = []
        current_text = 0
        current_vae = 0
        current_vit = 0
        
        for step in sequence_plan:
            step_type = step.get('type', 'unknown')
            loss = step.get('loss', 0)
            
            if step_type == 'vit_image':
                current_vit += 1
                flow_parts.append(f"ViT#{current_vit}")
            elif step_type == 'vae_image':
                current_vae += 1
                if loss == 0:
                    flow_parts.append(f"🖼️Input#{current_vae}")
                else:
                    flow_parts.append(f"🎯Output#{current_vae}")
            elif step_type == 'text':
                current_text += 1
                if loss == 0:
                    flow_parts.append(f"💬Instruction#{current_text}")
                else:
                    flow_parts.append(f"🤖Response#{current_text}")
        
        for i in range(0, len(flow_parts), 4):
            line_parts = flow_parts[i:i+4]
            print("   " + " → ".join(line_parts))
        
        loss_components = []
        no_loss_components = []
        
        for i, step in enumerate(sequence_plan):
            step_type = step.get('type', 'unknown')
            loss = step.get('loss', 0)
            
            if loss == 1:
                loss_components.append(f"Step {i+1} ({step_type})")
            else:
                no_loss_components.append(f"Step {i+1} ({step_type})")
        
        print(f"\n🎯 Training Target Analysis:")
        print(f"   Loss components: {', '.join(loss_components) if loss_components else 'None'}")
        print(f"   Condition components: {', '.join(no_loss_components) if no_loss_components else 'None'}")
        
        print("="*80)
        print("✅ Inspection completed!")
        print("="*80 + "\n")

    def debug_single_sample(self, json_line, image_dir, row_idx=0):
        """
        Debug single sample, parse and inspect data structure
        
        Args:
            json_line: JSON string
            image_dir: Image directory path
            row_idx: Row index for logging
        
        Returns:
            Parsed data dictionary
        """
        print(f"\n🐛 DEBUG MODE: Processing sample #{row_idx}")
        print("-" * 50)
        
        try:
            data_item = json.loads(json_line)
            print(f"✅ JSON parsed successfully")
            print(f"   Keys: {list(data_item.keys())}")
            
            is_valid, error_msg = self._validate_data_item(data_item)
            if not is_valid:
                print(f"❌ Validation failed: {error_msg}")
                return {}
            
            print(f"✅ Data validation passed")
            
            data = self.parse_row(json_line, image_dir, row_idx)
            
            if data:
                print(f"✅ Data parsing completed")
                self.inspect_data_structure(data, detailed=True, show_tensor_shapes=True)
            else:
                print(f"❌ Data parsing returned empty result")
            
            return data
            
        except Exception as e:
            print(f"❌ Debug failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return {}

    def __iter__(self):
        """"""
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
    
        if self.data_status is not None:
            print(f"--------------------->> self.data_status {self.data_status}")
            print(f"worker_id: {worker_id}")
            status = self.data_status[worker_id]
            if isinstance(status, (list, tuple)):
                if len(status) >= 2:
                    global_row_group_start_id = status[0]
                    row_start_id = status[1] + 1
                elif len(status) == 1:
                    global_row_group_start_id = 0
                    row_start_id = status[0] + 1
                else:
                    global_row_group_start_id = 0
                    row_start_id = 0
            else:
                global_row_group_start_id = 0
                row_start_id = int(status) + 1
        else:
            global_row_group_start_id = 0
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming medical data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            
            for row_idx, (json_line, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                data = self.parse_row(json_line, image_dir, row_idx)
                
                if len(data) == 0:
                    continue
                
                data['data_indexes'] = {
                    "data_indexes": [row_idx],
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }
                
                yield data
            
            row_start_id = 0
            print(f"Medical {self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


 