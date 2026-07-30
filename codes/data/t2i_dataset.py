# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 Unimedvl Team
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by Unimedvl Team.
# Modifications: Enhanced T2I dataset with improved JSONL support and medical image generation capabilities.

import io
import json
import pyarrow.parquet as pq
import random
import traceback
from PIL import Image

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from data_utils import pil_img2rgb
from distributed_iterable_dataset import DistributedIterableDataset
from parquet_utils import get_parquet_data_paths, init_arrow_pf_fs
from interleave_datasets.interleave_t2i_dataset import InterleavedBaseIterableDataset

Image.MAX_IMAGE_PIXELS = 20000000  


class T2IIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        return get_parquet_data_paths(data_dir_list, num_used_data)

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]

                        for row_idx, row in df.iterrows():
                            num_tokens = 0
                            try:
                                image_byte = row['image']
                                image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))
                                if image.width > 4096 or image.height > 4096:
                                    print("skip very large image …")
                                    continue
                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2

                            try:
                                caption_dict = row['captions']
                                caption_dict = json.loads(caption_dict)
                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue

                            caps_token = [self.tokenizer.encode(v) for _, v in caption_dict.items()]
                            if len(caps_token) == 0:
                                print(f'no caption in rg#{row_group_id}, {parquet_file_path}')
                                caption_token = self.tokenizer.encode(' ')
                            else:
                                caption_token = random.choice(caps_token)

                            sequence_plan, text_ids_list = [], []
                            text_ids = caption_token
                            num_tokens += len(caption_token)
                            text_ids_list.append(text_ids)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })
                        
                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sample = dict(
                                image_tensor_list=[image_tensor], 
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class T2IIterableDataset_Ver1(InterleavedBaseIterableDataset):
    """
    T2I Dataset that supports reading output_img from JSONL files.
    Similar to VLM dataset but for T2I tasks with output image generation.
    
    Expected JSONL structure:
    - 'captions': JSON string or dict with caption dictionary
    - 'output_img': list with output image info [{"path": "relative/path/to/image.jpg"}]
    - or 'image': byte data for backward compatibility
    Expected input data structure:
    Image information: input_img and output_img content
    Conversation information: message content, including gpt and human parts, can be continuous dialogue but usually one round
    Human part provides image instructions and descriptions, gpt provides simple pre-generation descriptions
    """
    
    def __init__(
        self, dataset_name, transform, tokenizer, jsonl_path_list,
        data_dir_list, num_used_data, image_dir_list=None,
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
        shuffle_lines=False, shuffle_seed=0,
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the output images
        num_used_data: list of number of sampled data points for each jsonl
        image_dir_list: list of image directories, if None use data_dir_list
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(
            jsonl_path_list, 
            data_dir_list, 
            num_used_data, 
            shuffle_lines, 
            shuffle_seed
        )
        self.set_epoch()
        self.shuffle_lines = shuffle_lines
        self.shuffle_seed = shuffle_seed

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
                import random
                rng = random.Random(shuffle_seed)
                rng.shuffle(raw_data)
            
            if num_data_point == 0:
                raw_data = raw_data
            else:
                raw_data = raw_data[:num_data_point]
            
            print(f"--->> data from {jsonl_path}:{len(raw_data)}")
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])
        
        return data_paths

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0
        
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            
            for row_idx, (json_line, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                try:
                    data_item = json.loads(json_line.strip())
                    # print(f"data_item: {data_item}")
                    # print(f"-----------------------------------------------")
                    
                    num_tokens = 0
                    
                    output_image_list = []
                    image_key = "output_img" if "output_img" in data_item and len(data_item["output_img"]) > 0 else "input_img"
                    
                    if image_key not in data_item and len(data_item) == 0:
                        print(f"No valid image found in row#{row_idx}")
                        continue
                    
                    if image_key in data_item and data_item[image_key] is not None:
                        try:
                            output_img_data = data_item[image_key]
                            if isinstance(output_img_data, str):
                                output_img_data = json.loads(output_img_data)
                            
                            if isinstance(output_img_data, list) and len(output_img_data) > 0:
                                for img_info in output_img_data:
                                    if isinstance(img_info, dict) and 'path' in img_info:
                                        rel_path = img_info['path'].lstrip(os.sep)
                                        image_path = os.path.join(image_dir, rel_path)
                                        
                                        if not os.path.exists(image_path):
                                            print(f'Image file not found: {image_path} in row#{row_idx}')
                                            continue  # Skip to the next image
                                            
                                        if not os.path.isfile(image_path):
                                            print(f'Path is not a file: {image_path} in row#{row_idx}')
                                            continue  # Skip to the next image
                                            
                                        try:
                                            with Image.open(image_path) as img:
                                                img.verify()
                                                
                                            with Image.open(image_path) as img:
                                                # Manually check for potential decompression bomb to avoid warnings and excessive memory usage.
                                                # Pillow's default limit is 89,478,485 pixels.
                                                max_pixels = getattr(Image, "MAX_IMAGE_PIXELS", 89478485)
                                                
                                                if img.width * img.height > max_pixels:
                                                    print(f"Skipping large image (>{max_pixels / 1e6:.1f}M pixels): {image_path} with size {img.size} in row#{row_idx}")
                                                    continue
                                                
                                                if img.width >= 4096 or img.height >= 4096:
                                                    print(f"Skipping large image (>4096x4096): {image_path} with size {img.size} in row#{row_idx}")
                                                    continue
                                                                                                # The image must be converted before the 'with' block exits.
                                                output_image = pil_img2rgb(img)
                                                output_image_list.append(output_image)
                                        except Exception as e:
                                            print(f'Error opening or processing image {image_path}: {e} in row#{row_idx}')
                                            continue  # Skip to the next image
                        except Exception as e:
                            print(f'Error processing image list data structure: {e} in row#{row_idx}')
                            continue
                    
                    if len(output_image_list) == 0:
                        print(f'No valid image found in row#{row_idx}')
                        continue

                    image_tensor_list = []
                    for output_image in output_image_list:
                        image_tensor = self.transform(output_image)
                        image_tensor_list.append(image_tensor)
                        height, width = image_tensor.shape[1:]
                        num_tokens += width * height // transform_stride ** 2

                    messages = []
                    try: 
                        if "message" in data_item:
                            for message in data_item["message"]:
                                if "from" in message and "value" in message:
                                    if message["from"] == "human":
                                        messages.append((message["value"], "human"))
                                    elif message["from"] == "gpt":
                                        messages.append((message["value"], "gpt"))
                    except Exception as e:
                        print(f'Error loading messages: {e} in row#{row_idx}')
                        continue
                    
                    if len(messages) == 0:
                        print(f'No valid messages found in row#{row_idx}')
                        continue
                    
                    sequence_plan, text_ids_list = [], []
                    
                    for message_text, role in messages:
                        text_ids = self.tokenizer.encode(message_text)
                        if len(text_ids) > 0:
                            num_tokens += len(text_ids)
                            text_ids_list.append(text_ids)

                            if role == "human":
                                sequence_plan.append({
                                    'type': 'text',
                                    'enable_cfg': 1,
                                    'loss': 0,
                                    'special_token_loss': 0,
                                    'special_token_label': None,
                                })
                            elif role == "gpt":
                                sequence_plan.append({
                                    'type': 'text',
                                    'enable_cfg': 0,
                                    'loss': 1,
                                    'special_token_loss': 0,
                                    'special_token_label': None,
                                })

    
                    for _ in image_tensor_list:           
                        sequence_plan.append({
                            'type': 'vae_image',
                            'enable_cfg': 0,
                            'loss': 1,
                            'special_token_loss': 0,
                            'special_token_label': None,
                        })

                    has_loss = [item['loss'] for item in sequence_plan]
                    if sum(has_loss) == 0:
                        print(f'No loss defined, skipped row#{row_idx}')
                        continue

                    sample = dict(
                        image_tensor_list=image_tensor_list, 
                        text_ids_list=text_ids_list,
                        num_tokens=num_tokens,
                        sequence_plan=sequence_plan,
                        data_indexes={
                            "data_indexes": row_idx,
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        }
                    )
                        
                    yield sample

                except Exception as e:
                    print(f'Error processing row: {e} in row#{row_idx}')
                    traceback.print_exc()
                    continue

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")

