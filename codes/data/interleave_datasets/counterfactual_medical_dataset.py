# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 Unimedvl Team
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by Unimedvl Team.
# Modifications: Enhanced counterfactual medical dataset with improved image processing and debugging features.

import json
import os
import traceback
from PIL import Image, ImageFile, PngImagePlugin

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

# Import base classes and utility functions from parent directory
from interleave_datasets.interleave_t2i_dataset import InterleavedBaseIterableDataset
from data_utils import pil_img2rgb
from distributed_iterable_dataset import DistributedIterableDataset

# Configure PIL image processing parameters for robustness
Image.MAX_IMAGE_PIXELS = 20000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class CounterfactualMedicalIterableDataset_ver1(InterleavedBaseIterableDataset):
    """
    Counterfactual Medical Image Dataset Loader (for CXR, CT, MRI, etc.)

    sequence_plan = [
        # Step 1: Original medical image VAE encoding (condition)
        {
            'type': 'vae_image',
            'enable_cfg': 1,        # Enable CFG as condition
            'loss': 0,              # No loss calculation
            'special_token_loss': 0,
            'special_token_label': None,
        },

        # Step 2: Original medical image ViT encoding (visual understanding)
        {
            'type': 'vit_image',
            'enable_cfg': 0,        # Disable CFG for condition
            'loss': 0,              # No loss calculation
            'special_token_loss': 0,
            'special_token_label': None,
        },

        # Step 3: Counterfactual instruction text (condition)
        {
            'type': 'text',
            'enable_cfg': 0,        # Disable CFG for condition
            'loss': 0,              # No loss calculation
            'special_token_loss': 0,
            'special_token_label': None,
        },

        # Step 4: Counterfactual medical image (target 1)
        {
            'type': 'vae_image',
            'enable_cfg': 0,        # Disable CFG for final output
            'loss': 1,              # Compute loss
            'special_token_loss': 0,
            'special_token_label': None,
        },

        # Step 5: Explanatory text (target 2)
        {
            'type': 'text',
            'enable_cfg': 0,        # DisEnable CFG
            'loss': 1,              # Compute loss
            'special_token_loss': 0,
            'special_token_label': None,
        }
    ]

    Data Format:
    {
        "main_task_type": "counterfactual_generation",
        "sub_task_type": "counterfactual_medical",
        "dataset": "counterfactual_medical_dataset",
        "input_img": [{"path": "/path/to/original_image.jpeg", "height": 512, "width": 512, ...}],
        "output_img": [{"path": "/path/to/counterfactual_image.jpeg", "height": 512, "width": 512, ...}],
        "message": [
            {"from": "human", "value": "What would this chest X-ray look like without pneumonia?"},
            {"from": "gpt", "value": "The generated counterfactual image shows normal lungs without inflammation."}
        ]
    }

    Sequence Construction Strategy:
    Original Image(cond) -> Instruction(cond) -> Counterfactual Image(target1) -> Explanation(target2)
    need_loss:  False    ->     False        ->          True               ->      True
    need_vae:   True     ->       -          ->          False              ->        -
    need_vit:   True     ->       -          ->          False              ->        -
    """
    
    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform,
        jsonl_path_list, data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        shuffle_lines=False, shuffle_seed=0,
    ):
        """
        Initialize Counterfactual Medical Dataset

        Args:
            dataset_name: Dataset name
            transform: VAE image transformation
            tokenizer: Text tokenizer
            vit_transform: ViT image transformation
            jsonl_path_list: List of JSONL file paths
            data_dir_list: List of image directories
            num_used_data: Number of samples to use from each source
            Other params: Distributed training parameters
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
        """Get and process data paths"""
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

            print(f"--->> Counterfactual medical data from {jsonl_path}: {len(raw_data)}")
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])

        return data_paths

    def _validate_data_item(self, data_item):
        """Validate data item completeness"""
        # Check required fields
        if not data_item.get("input_img"):
            return False, "Missing input_img"
        if not data_item.get("output_img"):
            return False, "Missing output_img"
        if not data_item.get("message"):
            return False, "Missing message"

        # Check image paths
        if not isinstance(data_item["input_img"], list) or len(data_item["input_img"]) == 0:
            return False, "Invalid input_img format"
        if not isinstance(data_item["output_img"], list) or len(data_item["output_img"]) == 0:
            return False, "Invalid output_img format"

        # Check message format
        if not isinstance(data_item["message"], list) or len(data_item["message"]) < 2:
            return False, "Invalid message format"

        return True, ""

    def _extract_messages(self, message_list):
        """Extract human and gpt messages from message list"""
        human_message = ""
        gpt_message = ""

        for msg in message_list:
            if msg.get("from") == "human":
                human_message = msg.get("value", "")
            elif msg.get("from") == "gpt":
                gpt_message = msg.get("value", "")

        return human_message, gpt_message

    def _process_text_with_image_tags(self, text, num_images=1):
        """
        Process text containing <image> tags, splitting into text and image placeholders

        Args:
            text: Text containing <image> tags
            num_images: Expected number of images

        Returns:
            elements: List containing text and image elements
        """
        elements = []

        if '<image>' not in text:
            # If no image tags, add text directly
            if text.strip():
                elements.append({
                    'type': 'text',
                    'content': text.strip(),
                })
        else:
            # Split text by <image> tags
            text_list = text.split('<image>')
            image_count = 0

            for idx, text_part in enumerate(text_list):
                # Add non-empty text fragments
                if text_part.strip():
                    elements.append({
                        'type': 'text',
                        'content': text_part.strip(),
                    })

                # Add image placeholder between text fragments
                if (idx != len(text_list) - 1) and (image_count < num_images):
                    elements.append({
                        'type': 'image',
                        'image_idx': image_count,
                    })
                    image_count += 1

        return elements

    def _load_image_safely(self, image_path, row_idx):
        """Safely load image with existence and size checks"""
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f'Medical image file not found: {image_path} in row#{row_idx}')
            return None

        # Check if path is a valid file
        if not os.path.isfile(image_path):
            print(f'Path is not a file: {image_path} in row#{row_idx}')
            return None

        try:
            with Image.open(image_path) as img:
                # Verify image can be loaded properly
                img.verify()

            # Reopen image for processing (verify closes the file)
            with Image.open(image_path) as img:
                # Check image dimensions
                max_pixels = getattr(Image, "MAX_IMAGE_PIXELS", 89478485)
                if img.width * img.height > max_pixels:
                    print(f"Skipping large medical image (>{max_pixels / 1e6:.1f}M pixels): {image_path} with size {img.size} in row#{row_idx}")
                    return None
                if img.width >= 4096 or img.height >= 4096:
                    print(f"Skipping large medical image (>4096x4096): {image_path} with size {img.size} in row#{row_idx}")
                    return None

                # Convert to RGB format
                return pil_img2rgb(img.copy())
        except Exception as e:
            print(f"Error loading medical image {image_path}: {e} in row#{row_idx}")
            return None

    def parse_row(self, json_line, image_dir, row_idx):
        """
        Parse a single data row and build sequence

        Sequence Construction Logic:
        1. Original medical image as condition (need_loss=False, need_vae=True, need_vit=True)
        2. Counterfactual instruction as condition (need_loss=False)
        3. Counterfactual medical image as target (need_loss=True, need_vae=False, need_vit=False)
        4. Explanatory text as target (need_loss=True)
        """
        try:
            # Parse JSON data
            data_item = json.loads(json_line)

            # Validate data completeness
            is_valid, error_msg = self._validate_data_item(data_item)
            if not is_valid:
                print(f"\n❌ Data validation failed in row#{row_idx}: {error_msg}")
                return {}

            # Initialize sequence data
            data = self._init_data()

            # 1. Load and add original medical image as condition
            input_img_info = data_item["input_img"][0]
            input_img_rel_path = input_img_info["path"].lstrip(os.sep)
            input_img_path = os.path.join(image_dir, input_img_rel_path)

            input_image = self._load_image_safely(input_img_path, row_idx)
            if input_image is None:
                print(f"\n❌ Failed to load input medical image: {input_img_path} in row#{row_idx}")
                return {}

            # Add original medical image: as condition, needs VAE and ViT encoding
            data = self._add_image(
                data,
                input_image,
                need_loss=False,  # No loss calculation
                need_vae=True,    # VAE encoding for diffusion condition
                need_vit=True,    # ViT encoding for visual understanding
                enable_cfg=0      # Enable CFG as condition
            )

            # 2. Extract and process message text (handle <image> tags)
            human_instruction, gpt_response = self._extract_messages(data_item["message"])

            if not human_instruction.strip():
                print(f"\n❌ Empty counterfactual instruction in row#{row_idx}")
                return {}

            # Process human instruction containing <image> tags
            human_elements = self._process_text_with_image_tags(human_instruction, num_images=1)

            for element in human_elements:
                if element['type'] == 'text':
                    # Add text part: as condition, no loss calculation
                    data = self._add_text(data, element['content'], need_loss=False, enable_cfg=0)
                elif element['type'] == 'image':
                    # <image> tag corresponds to input image, already added in step 1
                    pass

            # 3. Load and add counterfactual medical image as first target
            output_img_info = data_item["output_img"][0]
            output_img_rel_path = output_img_info["path"].lstrip(os.sep)
            output_img_path = os.path.join(image_dir, output_img_rel_path)

            output_image = self._load_image_safely(output_img_path, row_idx)
            if output_image is None:
                print(f"\n❌ Failed to load counterfactual medical image: {output_img_path} in row#{row_idx}")
                return {}

            # Add counterfactual medical image: as first target
            data = self._add_image(
                data,
                output_image,
                need_loss=True,   # Compute loss
                need_vae=False,   # No VAE encoding (endpoint)
                need_vit=False,   # No ViT encoding (endpoint)
                enable_cfg=0      # Output target doesn't allow dropout
            )

            # 4. Add explanatory text as second target
            if gpt_response.strip():
                data = self._add_text(data, gpt_response.strip(), need_loss=True, enable_cfg=0)

            return data

        except Exception as e:
            print(f"\n❌ Error processing counterfactual medical data row#{row_idx}: {e}")
            print(f"Exception type: {type(e).__name__}")
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
        print("🔍 COUNTERFACTUAL Medical Image Data Structure Inspection")
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
            print(f"   {'-'*4} {'-'*12} {'-'*4} {'-'*3} {'-'*3} {'-'*3} {'-'*30}")
            
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
                        desc = f"Original Medical #{vae_count} (Condition)"
                    elif loss == 1 and cfg == 0:
                        desc = f"Counterfactual Medical #{vae_count} (Target)"
                    else:
                        desc = f"VAE Image #{vae_count}"
                elif step_type == 'vit_image':
                    vit_count += 1
                    desc = f"ViT Encoding #{vit_count} (Vision)"
                elif step_type == 'text':
                    text_count += 1
                    if loss == 0:
                        desc = f"Counterfactual Instruction #{text_count}"
                    else:
                        desc = f"Explanation Text #{text_count} (Target)"
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
            print(f"🩻 Medical Image Tensor Analysis:")
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
        
        print(f"\n🔄 Counterfactual Medical Image Sequence Flow:")
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
                    flow_parts.append(f"🩻Original#{current_vae}")
                else:
                    flow_parts.append(f"🎯Target#{current_vae}")
            elif step_type == 'text':
                current_text += 1
                if loss == 0:
                    flow_parts.append(f"💭Instruction#{current_text}")
                else:
                    flow_parts.append(f"📝Explanation#{current_text}")
        
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
        print("✅ Counterfactual Medical Image Inspection Complete!")
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
        print(f"\n🐛 Debug Mode: Processing counterfactual medical sample #{row_idx}")
        print("-" * 60)
        
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
        """Data iterator with token limit filtering"""
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()

        # Token limit for memory safety (configurable)
        MAX_TOKENS_PER_SAMPLE = getattr(self, 'max_tokens_per_sample', 18000)

        if self.data_status is not None:
            status = self.data_status[worker_id]
            if isinstance(status, (list, tuple)):
                if len(status) >= 2:  # Parquet case
                    global_row_group_start_id = status[0]
                    row_start_id = status[1] + 1
                elif len(status) == 1:  # JSONL case
                    global_row_group_start_id = 0
                    row_start_id = status[0] + 1
                else:
                    global_row_group_start_id = 0
                    row_start_id = 0
            else:  # Scalar or other format
                global_row_group_start_id = 0
                row_start_id = int(status) + 1
        else:
            global_row_group_start_id = 0
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"Resuming counterfactual medical data from row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]

            for row_idx, (json_line, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                # Parse single row
                data = self.parse_row(json_line, image_dir, row_idx)

                # Skip empty data
                if len(data) == 0:
                    continue

                # Check token limit to prevent OOM
                num_tokens = data.get('num_tokens', 0)
                if num_tokens > MAX_TOKENS_PER_SAMPLE:
                    print(f"⚠️ Skipping oversized sample at row#{row_idx}: {num_tokens} tokens > {MAX_TOKENS_PER_SAMPLE} limit")
                    continue

                # Add data index information
                data['data_indexes'] = {
                    "data_indexes": [row_idx],
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }

                yield data

            # Reset row index for next epoch
            row_start_id = 0
            print(f"Counterfactual medical dataset {self.dataset_name} repeating in rank-{self.local_rank} worker-{worker_id}")

