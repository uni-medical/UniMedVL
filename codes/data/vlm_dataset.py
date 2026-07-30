# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 Unimedvl Team
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by Unimedvl Team.
# Modifications: Enhanced VLM dataset with visual reconstruction capabilities and text-only dataset support.

# Import json module for handling JSON data
import json
# Import os module for OS-related functions like path handling
import os
# Import traceback module for printing exception information
import traceback
# Import Image, ImageFile, PngImagePlugin from PIL for image processing
from PIL import Image, ImageFile, PngImagePlugin
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import pil_img2rgb function from local data_utils module for PIL image to RGB conversion
from data_utils import pil_img2rgb
# Import DistributedIterableDataset class from local distributed_iterable_dataset module
from distributed_iterable_dataset import DistributedIterableDataset
# Import InterleavedBaseIterableDataset class
from interleave_datasets.interleave_t2i_dataset import InterleavedBaseIterableDataset

# Configure Pillow library's maximum image pixels to prevent DecompressionBombError for large images
Image.MAX_IMAGE_PIXELS = 20000000
# Configure Pillow library to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Define decompression size for subsequent PNG text chunk size setting
MaximumDecompressedSize = 1024
# Define 1MB size in bytes
MegaByte = 2 ** 20
# Configure PNG image plugin's maximum text chunk size to avoid issues with oversized text chunks
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

# class SftJSONLIterableDataset_with_VisualReconstruction_Ver2(DistributedIterableDataset,InterleavedBaseIterableDataset):
class SftJSONLIterableDataset_with_VisualReconstruction_Ver2(InterleavedBaseIterableDataset):
    
    """
    sequence_plan = [
        {'type': 'vae_image', 'loss': 0, 'enable_cfg': 1},  # img1 latent
        {'type': 'vit_image', 'loss': 0, 'enable_cfg': 1}, # img1 patch

        {'type': 'text', 'loss': 0, 'enable_cfg': 0},      # text1

        {'type': 'vae_image', 'loss': 0, 'enable_cfg': 1},  # img2 latent
        {'type': 'vit_image', 'loss': 0, 'enable_cfg': 1}, # img2 patch

        {'type': 'text', 'loss': 0, 'enable_cfg': 0},      # text2

        {'type': 'text', 'loss': 1, 'enable_cfg': 0},      # answer text A

        {'type': 'vae_image', 'loss': 1, 'enable_cfg': 0}, # rec img1 latent 
        {'type': 'vae_image', 'loss': 1, 'enable_cfg': 0}, # rec img2 latent 
    ]

    """
    
    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform,
        jsonl_path_list, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
        shuffle_lines=False, shuffle_seed=0,
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
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
        # print(f"data_paths: {self.data_paths}")
        self.set_epoch()

    def get_data_paths(
        self, 
        jsonl_path_list, 
        data_dir_list, 
        num_used_data, 
        shuffle_lines, 
        shuffle_seed,
    ):
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
            print(f"--->> data from {jsonl_path}:{len(raw_data)}")
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])
        return data_paths

    def change_format(self, data, num_images):
        
        elements = []
        for conversation in data['message']:
            if conversation['from'] == 'human':
                if '<image>' not in conversation['value']:
                    elements.append({
                        'type': 'text',
                        'has_loss': 0,
                        'text': conversation['value'] + "Please also reconstruct the image at the end",
                    })
                else:
                    text_list = conversation['value'].split('<image>')
                    last_non_empty_idx = -1
                    for i, text in enumerate(text_list):
                        if text.strip() != '':
                            last_non_empty_idx = i
                    
                    for idx, text in enumerate(text_list):
                        if text.strip() != '':
                            text_content = text.strip()
                            if idx == last_non_empty_idx:
                                text_content += " Please also reconstruct the image at the end"
                            elements.append({
                                'type': 'text',
                                'has_loss': 0,
                                'text': text_content,
                            })
                        if (idx != len(text_list) - 1) and (idx < num_images):
                            elements.append({'type': 'image', "image_index": idx})
                            
            elif conversation['from'] == 'gpt':
                elements.append({
                    'type': 'text',
                    'has_loss': 1,
                    'text': conversation['value'],
                })
        return elements

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (json_line, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                data = self._init_data()

                try:
                    data_item = json.loads(json_line)
                    raw_images = []
                    skip_row = False
                    if 'input_img' in data_item:
                        image_infos = data_item['input_img']
                        if not isinstance(image_infos, list):
                            image_infos = [image_infos]
                        
                        for image_info in image_infos:
                            rel = image_info["path"].lstrip(os.sep)
                            image_path = os.path.join(image_dir, rel)
                            
                            if not os.path.exists(image_path):
                                print(f'Image file not found: {image_path} in row#{row_idx}')
                                skip_row = True
                                break
                                
                            if not os.path.isfile(image_path):
                                print(f'Path is not a file: {image_path} in row#{row_idx}')
                                skip_row = True
                                break
                                
                            try:
                                with Image.open(image_path) as img:
                                    img.verify()
                                    
                                with Image.open(image_path) as img:
                                    max_pixels = getattr(Image, "MAX_IMAGE_PIXELS", 89478485)
                                    if img.width * img.height > max_pixels:
                                        print(f"Skipping large image (>{max_pixels / 1e6:.1f}M pixels): {image_path} with size {img.size} in row#{row_idx}")
                                        skip_row = True
                                        break
                                    if img.width >= 4096 or img.height >= 4096:
                                        print(f"Skipping large image (>4096x4096): {image_path} with size {img.size} in row#{row_idx}")
                                        skip_row = True
                                        break
                                    raw_images.append(pil_img2rgb(img))
                            except Exception as e:
                                print(f'Error opening or processing image {image_path}: {e} in row#{row_idx}')
                                skip_row = True
                                break
                        
                        if skip_row:
                            continue
                except:
                    traceback.print_exc()
                    continue
                
                elements = self.change_format(data_item, len(raw_images))

                for item in elements:
                    if item['type'] == 'text':
                        text_data = item['text']
                        if len(self.tokenizer.encode(text_data)) > 0:
                            data = self._add_text(
                                data,
                                text_data,
                                need_loss=item['has_loss'],
                                enable_cfg=False,
                            )
                    elif item['type'] == 'image':
                        image = raw_images[item['image_index']]
                        data = self._add_image(
                            data, 
                            image,
                            need_loss=False, 
                            need_vae=True,  
                            need_vit=True, 
                            enable_cfg=False
                        )
                        
                # Add VAE reconstruction for each input image, which corresponds to the VAE input.
                for image in raw_images:
                    data = self._add_image(
                        data,
                        image,
                        need_loss=True,
                        need_vae=False,
                        need_vit=False,
                    )

            
                has_loss = [item['loss'] for item in data['sequence_plan']]
                if sum(has_loss) == 0:
                    print(f'No loss defined, skipped.')
                    continue

                data['data_indexes'] = {
                    "data_indexes": row_idx,
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }
                
                # print(f"--------------------------------")
                # print(f"data: {data.keys()}")
                # print(f"data sequence: {data['sequence_plan']}")
                # print("--------------------------------")
                
                yield data
            
            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


# class SftJSONLIterableDataset_with_VisualReconstruction_Ver1(DistributedIterableDataset,InterleavedBaseIterableDataset):
class SftJSONLIterableDataset_with_VisualReconstruction_Ver1(InterleavedBaseIterableDataset):
    
    """
    sequence_plan = [
        {'type': 'vae_image', 'loss': 0, 'enable_cfg': 1},  # img1 latent
        {'type': 'vit_image', 'loss': 0, 'enable_cfg': 1}, # img1 patch

        {'type': 'text', 'loss': 0, 'enable_cfg': 0},      # text1

        {'type': 'vae_image', 'loss': 0, 'enable_cfg': 1},  # img2 latent
        {'type': 'vit_image', 'loss': 0, 'enable_cfg': 1}, # img2 patch

        {'type': 'text', 'loss': 0, 'enable_cfg': 0},      # text2

        {'type': 'text', 'loss': 1, 'enable_cfg': 0},      # answer text A

        {'type': 'vae_image', 'loss': 1, 'enable_cfg': 0}, # rec img1 latent 
        {'type': 'vae_image', 'loss': 1, 'enable_cfg': 0}, # rec img2 latent 
    ]

    """
    
    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform,
        jsonl_path_list, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
        shuffle_lines=False, shuffle_seed=0,
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
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
        # print(f"data_paths: {self.data_paths}")
        self.set_epoch()

    def get_data_paths(
        self, 
        jsonl_path_list, 
        data_dir_list, 
        num_used_data, 
        shuffle_lines, 
        shuffle_seed,
    ):
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
            print(f"--->> data from {jsonl_path}:{len(raw_data)}")
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])
        return data_paths

    def change_format(self, data, num_images):
        
        elements = []
        for conversation in data['message']:
            if conversation['from'] == 'human':
                if '<image>' not in conversation['value']:
                    elements.append({
                        'type': 'text',
                        'has_loss': 0,
                        'text': conversation['value'] + "Please also reconstruct the image at the end",
                    })
                else:
                    text_list = conversation['value'].split('<image>')
                    last_non_empty_idx = -1
                    for i, text in enumerate(text_list):
                        if text.strip() != '':
                            last_non_empty_idx = i
                    
                    for idx, text in enumerate(text_list):
                        if text.strip() != '':
                            text_content = text.strip()
                            if idx == last_non_empty_idx:
                                text_content += " Please also reconstruct the image at the end"
                            elements.append({
                                'type': 'text',
                                'has_loss': 0,
                                'text': text_content,
                            })
                        if (idx != len(text_list) - 1) and (idx < num_images):
                            elements.append({'type': 'image', "image_index": idx})
                            
            elif conversation['from'] == 'gpt':
                elements.append({
                    'type': 'text',
                    'has_loss': 1,
                    'text': conversation['value'],
                })
        return elements

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (json_line, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                data = self._init_data()

                try:
                    data_item = json.loads(json_line)
                    raw_images = []
                    skip_row = False
                    if 'input_img' in data_item:
                        image_infos = data_item['input_img']
                        if not isinstance(image_infos, list):
                            image_infos = [image_infos]
                        
                        for image_info in image_infos:
                            rel = image_info["path"].lstrip(os.sep)
                            image_path = os.path.join(image_dir, rel)
                            
                            if not os.path.exists(image_path):
                                print(f'Image file not found: {image_path} in row#{row_idx}')
                                skip_row = True
                                break
                                
                            if not os.path.isfile(image_path):
                                print(f'Path is not a file: {image_path} in row#{row_idx}')
                                skip_row = True
                                break
                                
                            try:
                                with Image.open(image_path) as img:
                                    img.verify()
                                    
                                with Image.open(image_path) as img:
                                    max_pixels = getattr(Image, "MAX_IMAGE_PIXELS", 89478485)
                                    if img.width * img.height > max_pixels:
                                        print(f"Skipping large image (>{max_pixels / 1e6:.1f}M pixels): {image_path} with size {img.size} in row#{row_idx}")
                                        skip_row = True
                                        break
                                    if img.width >= 4096 or img.height >= 4096:
                                        print(f"Skipping large image (>4096x4096): {image_path} with size {img.size} in row#{row_idx}")
                                        skip_row = True
                                        break
                                    raw_images.append(pil_img2rgb(img))
                            except Exception as e:
                                print(f'Error opening or processing image {image_path}: {e} in row#{row_idx}')
                                skip_row = True
                                break
                        
                        if skip_row:
                            continue
                except:
                    traceback.print_exc()
                    continue
                
                elements = self.change_format(data_item, len(raw_images))

                for item in elements:
                    if item['type'] == 'text':
                        text_data = item['text']
                        if len(self.tokenizer.encode(text_data)) > 0:
                            data = self._add_text(
                                data,
                                text_data,
                                need_loss=item['has_loss'],
                                enable_cfg=False,
                            )
                    elif item['type'] == 'image':
                        image = raw_images[item['image_index']]
                        data = self._add_image(
                            data, 
                            image,
                            need_loss=False, 
                            need_vae=False,
                            need_vit=True, 
                            enable_cfg=False
                        )
                        
                # Add VAE reconstruction for each input image, which corresponds to the VAE input.
                for image in raw_images:
                    data = self._add_image(
                        data,
                        image,
                        need_loss=True,
                        need_vae=False,
                        need_vit=False,
                    )

            
                has_loss = [item['loss'] for item in data['sequence_plan']]
                if sum(has_loss) == 0:
                    print(f'No loss defined, skipped.')
                    continue

                data['data_indexes'] = {
                    "data_indexes": row_idx,
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }
                
                # print(f"--------------------------------")
                # print(f"data: {data.keys()}")
                # print(f"data sequence: {data['sequence_plan']}")
                # print("--------------------------------")
                
                yield data
            
            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


# class SftJSONLIterableDataset_with_VisualReconstruction_Ver1(DistributedIterableDataset,InterleavedBaseIterableDataset):
class SftJSONLIterableDataset_Ver1(InterleavedBaseIterableDataset):
    
    """
    sequence_plan = [
        {'type': 'vit_image', 'loss': 0, 'enable_cfg': 1}, # img1 patch

        {'type': 'text', 'loss': 0, 'enable_cfg':0},      # text1

        {'type': 'vit_image', 'loss': 0, 'enable_cfg': 1}, # img2 patch

        {'type': 'text', 'loss': 0, 'enable_cfg': 0},      # text2

        {'type': 'text', 'loss': 1, 'enable_cfg': 0},      # answer text A
    ]

    """
    
    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform,
        jsonl_path_list, data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
        shuffle_lines=False, shuffle_seed=0,
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
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
        # print(f"data_paths: {self.data_paths}")
        self.set_epoch()

    def get_data_paths(
        self, 
        jsonl_path_list, 
        data_dir_list, 
        num_used_data, 
        shuffle_lines, 
        shuffle_seed,
    ):
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
            print(f"--->> data from {jsonl_path}:{len(raw_data)}")
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])
        return data_paths

    def change_format(self, data, num_images):
        
        elements = []
        for conversation in data['message']:
            if conversation['from'] == 'human':
                if '<image>' not in conversation['value']:
                    elements.append({
                        'type': 'text',
                        'has_loss': 0,
                        'text': conversation['value'],
                    })
                else:
                    text_list = conversation['value'].split('<image>')
                    for idx, text in enumerate(text_list):
                        if text.strip() != '':
                            elements.append({
                                'type': 'text',
                                'has_loss': 0,
                                'text': text.strip(),
                            })
                        if (idx != len(text_list) - 1) and (idx < num_images):
                            elements.append({'type': 'image', "image_index": idx})
            elif conversation['from'] == 'gpt':
                elements.append({
                    'type': 'text',
                    'has_loss': 1,
                    'text': conversation['value'],
                })
        return elements

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (json_line, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                data = self._init_data()

                try:
                    data_item = json.loads(json_line)
                    raw_images = []
                    skip_row = False
                    if 'input_img' in data_item:
                        image_infos = data_item['input_img']
                        if not isinstance(image_infos, list):
                            image_infos = [image_infos]

                        for image_info in image_infos:
                            rel = image_info["path"].lstrip(os.sep)
                            image_path = os.path.join(image_dir, rel)
                            
                            if not os.path.exists(image_path):
                                print(f'Image file not found: {image_path} in row#{row_idx}')
                                skip_row = True
                                break
                                
                            if not os.path.isfile(image_path):
                                print(f'Path is not a file: {image_path} in row#{row_idx}')
                                skip_row = True
                                break
                                
                            try:
                                with Image.open(image_path) as img:
                                    img.verify()
                                    
                                with Image.open(image_path) as img:
                                    max_pixels = getattr(Image, "MAX_IMAGE_PIXELS", 89478485)
                                    if img.width * img.height > max_pixels:
                                        print(f"Skipping large image (>{max_pixels / 1e6:.1f}M pixels): {image_path} with size {img.size} in row#{row_idx}")
                                        skip_row = True
                                        break
                                    if img.width >= 4096 or img.height >= 4096:
                                        print(f"Skipping large image (>4096x4096): {image_path} with size {img.size} in row#{row_idx}")
                                        skip_row = True
                                        break
                                    raw_images.append(pil_img2rgb(img))
                            except Exception as e:
                                print(f'Error opening or processing image {image_path}: {e} in row#{row_idx}')
                                skip_row = True
                                break
                        
                        if skip_row:
                            continue
                except:
                    traceback.print_exc()
                    continue
                
                elements = self.change_format(data_item, len(raw_images))

                for item in elements:
                    if item['type'] == 'text':
                        text_data = item['text']
                        if len(self.tokenizer.encode(text_data)) > 0:
                            data = self._add_text(
                                data,
                                text_data,
                                need_loss=item['has_loss'],
                                enable_cfg=False,
                            )
                    elif item['type'] == 'image':
                        image = raw_images[item['image_index']]
                        data = self._add_image(
                            data, 
                            image,
                            need_loss=False, 
                            need_vae=False, 
                            need_vit=True, 
                            enable_cfg=True
                        )
                        
                # # Add VAE reconstruction for each input image, which corresponds to the VAE input.
                # for image in raw_images:
                #     data = self._add_image(
                #         data,
                #         image,
                #         need_loss=True,
                #         need_vae=False,
                #         need_vit=False,
                #     )

            
                has_loss = [item['loss'] for item in data['sequence_plan']]
                if sum(has_loss) == 0:
                    print(f'No loss defined, skipped.')
                    continue

                data['data_indexes'] = {
                    "data_indexes": row_idx,
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }
                
                # print(f"--------------------------------")
                # print(f"data: {data.keys()}")
                # print(f"data sequence: {data['sequence_plan']}")
                # print("--------------------------------")
                
                yield data
            
            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class SftJSONLIterableDataset_TextOnly(InterleavedBaseIterableDataset):
    """
    Text-only dataset loader for pure text conversations, based on SftJSONLIterableDataset_Ver1
    
    sequence_plan = [
        {'type': 'text', 'loss': 0, 'enable_cfg': 0},      # user input text
        
        {'type': 'text', 'loss': 1, 'enable_cfg': 0},      # assistant response text
    ]
    """
    
    def __init__(
        self, dataset_name, tokenizer, 
        jsonl_path_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
        shuffle_lines=False, shuffle_seed=0, data_dir_list = None
    ):
        """
        jsonl_path_list: list of jsonl file paths
        num_used_data: list of number of sampled data points for each jsonl
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_dir_list = data_dir_list
        self.data_paths = self.get_data_paths(
            jsonl_path_list, 
            num_used_data, 
            shuffle_lines, 
            shuffle_seed,
        )
        self.set_epoch()

    def get_data_paths(
        self, 
        jsonl_path_list, 
        num_used_data, 
        shuffle_lines, 
        shuffle_seed,
    ):
        data_paths = []
        for jsonl_path, num_data_point in zip(jsonl_path_list, num_used_data):
            with open(jsonl_path, 'r') as f:
                raw_data = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            if num_data_point == 0:
                raw_data = raw_data
            else:
                raw_data = raw_data[:num_data_point]
            print(f"--->> text-only data from {jsonl_path}:{len(raw_data)}")
            data_paths.extend([json_data for json_data in raw_data])
        return data_paths

    def change_format(self, data):
        elements = []
        for conversation in data['message']:
            if conversation['from'] == 'human':
                text_content = conversation['value']
                cleaned_text = text_content.replace('<image>', '').strip()
                cleaned_text = ' '.join(cleaned_text.split())
                
                if cleaned_text:
                    elements.append({
                        'type': 'text',
                        'has_loss': 0,
                        'text': cleaned_text,
                    })
            elif conversation['from'] == 'gpt':
                text_content = conversation['value']
                cleaned_text = text_content.replace('<image>', '').strip()
                cleaned_text = ' '.join(cleaned_text.split())
                
                if cleaned_text:
                    elements.append({
                        'type': 'text',
                        'has_loss': 1,
                        'text': cleaned_text,
                    })
        return elements

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming text-only data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, json_line in enumerate(data_paths_per_worker_, start=row_start_id):
                data = self._init_data()

                try:
                    data_item = json.loads(json_line)
                except:
                    traceback.print_exc()
                    continue
                
                elements = self.change_format(data_item)

                for item in elements:
                    if item['type'] == 'text':
                        text_data = item['text']
                        if len(self.tokenizer.encode(text_data)) > 0:
                            data = self._add_text(
                                data,
                                text_data,
                                need_loss=item['has_loss'],
                                enable_cfg=False,
                            )

                has_loss = [item['loss'] for item in data['sequence_plan']]
                if sum(has_loss) == 0:
                    print(f'No loss defined, skipped.')
                    continue

                data['data_indexes'] = {
                    "data_indexes": row_idx,
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }
                
                yield data
            
            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
