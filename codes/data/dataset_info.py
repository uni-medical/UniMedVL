# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 UniMedVL Team
# SPDX-License-Identifier: Apache-2.0
#
# Dataset registry for UniMedVL training / fine-tuning.
# Replace every 'path_to_your_xxx' placeholder with your actual data paths.
# Top-level keys of DATASET_INFO match DATASET_REGISTRY and the top-level keys
# of the dataset config YAML (see data/configs/example.yaml).
# JSONL loaders expect 'data_dir' + 'jsonl_path'; parquet loaders expect
# 'data_dir' + 'num_files'.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from interleave_datasets import MedicalImageEditingIterableDataset_ver1, CounterfactualMedicalIterableDataset_ver1
from t2i_dataset import T2IIterableDataset, T2IIterableDataset_Ver1
from vlm_dataset import SftJSONLIterableDataset_with_VisualReconstruction_Ver1, SftJSONLIterableDataset_with_VisualReconstruction_Ver2, SftJSONLIterableDataset_Ver1, SftJSONLIterableDataset_TextOnly


DATASET_REGISTRY = {
    # Image-text conversation SFT
    'SftJSONLIterableDataset_Ver1': SftJSONLIterableDataset_Ver1,
    'SftJSONLIterableDataset_TextOnly': SftJSONLIterableDataset_TextOnly,
    'SftJSONLIterableDataset_with_VisualReconstruction_Ver1': SftJSONLIterableDataset_with_VisualReconstruction_Ver1,
    'SftJSONLIterableDataset_with_VisualReconstruction_Ver2': SftJSONLIterableDataset_with_VisualReconstruction_Ver2,
    # Text-to-image generation
    't2i_pretrain': T2IIterableDataset,
    'T2IIterableDataset_Ver1': T2IIterableDataset_Ver1,
    # Interleaved image-text editing / generation
    'MedicalImageEditingIterableDataset_ver1': MedicalImageEditingIterableDataset_ver1,
    'CounterfactualMedicalIterableDataset_ver1': CounterfactualMedicalIterableDataset_ver1,
}

DATASET_INFO = {
    'SftJSONLIterableDataset_Ver1': {
        'medical_vqa_sft': {
            'data_dir': 'path_to_your_medical_images',
            'jsonl_path': 'path_to_your_medical_annotations.jsonl',
        },
    },
    'SftJSONLIterableDataset_TextOnly': {
        'medical_text_sft': {
            'jsonl_path': 'path_to_your_text_only_data.jsonl',
        },
    },
    'SftJSONLIterableDataset_with_VisualReconstruction_Ver1': {
        'medical_vqa_recon_v1': {
            'data_dir': 'path_to_your_medical_images',
            'jsonl_path': 'path_to_your_medical_annotations.jsonl',
        },
    },
    'SftJSONLIterableDataset_with_VisualReconstruction_Ver2': {
        'medical_vqa_recon_v2': {
            'data_dir': 'path_to_your_medical_images',
            'jsonl_path': 'path_to_your_medical_annotations.jsonl',
        },
    },
    't2i_pretrain': {
        'medical_t2i_parquet': {
            'data_dir': 'path_to_your_t2i_parquet_data',
            'num_files': 10,
        },
    },
    'T2IIterableDataset_Ver1': {
        'medical_t2i': {
            'data_dir': 'path_to_your_generation_images',
            'jsonl_path': 'path_to_your_generation_annotations.jsonl',
        },
    },
    'MedicalImageEditingIterableDataset_ver1': {
        'medical_image_editing': {
            'data_dir': 'path_to_your_medical_editing_images',
            'jsonl_path': 'path_to_your_medical_editing_annotations.jsonl',
        },
    },
    'CounterfactualMedicalIterableDataset_ver1': {
        'counterfactual_cxr': {
            'data_dir': 'path_to_your_counterfactual_images',
            'jsonl_path': 'path_to_your_counterfactual_annotations.jsonl',
        },
    },
}
