# Training / Fine-tuning

We provide an example fine-tuning entry so you can adapt UniMedVL to your own data starting from the released checkpoint. The full multi-stage training recipes follow the curriculum described in the paper.

## 1. Install training dependencies

In addition to the inference environment, training requires `flash-attn` (install a build matching your PyTorch/CUDA versions):

```bash
pip install --no-build-isolation flash-attn
pip install pyyaml tensorboard
# Optional, only for video datasets:
pip install decord
```

## 2. Download the checkpoint

Same as inference:

```bash
huggingface-cli download General-Medical-AI/UniMedVL --local-dir ./checkpoints/UniMedVL
```

## 3. Register your datasets

Replace the `path_to_your_...` placeholders in `codes/data/dataset_info.py` with your data paths, then configure the dataset mixture in `codes/data/configs/example.yaml`. Three loader families are provided:

| YAML key | Loader | Data format |
|---|---|---|
| `SftJSONLIterableDataset_Ver1` (also `_TextOnly`, `_with_VisualReconstruction_Ver1/Ver2`) | Image-text conversation SFT | `data_dir` + `jsonl_path` |
| `MedicalImageEditingIterableDataset_ver1`, `CounterfactualMedicalIterableDataset_ver1` | Interleaved image-text editing | `data_dir` + `jsonl_path` |
| `T2IIterableDataset_Ver1` (jsonl), `t2i_pretrain` (parquet) | Text-to-image generation | jsonl / parquet |

Each JSONL line contains a `message` conversation list (with `<image>` placeholders) and optional `input_img` / `output_img` entries.

## 4. Launch fine-tuning

Single node, FSDP:

```bash
cd codes
MODEL_PATH=./checkpoints/UniMedVL NUM_GPUS=8 bash scripts/train_example.sh
```

Training logs are written to TensorBoard under the output directory (`tensorboard --logdir output/finetune_example`). Key flags such as `--layer_module Qwen2MoTDecoderLayer` and `--max_latent_size 64` must stay consistent with the released checkpoint configuration. Adjust `--freeze_llm/--freeze_vit/--freeze_vae` and the token budget to fit your hardware.

Note: with `--visual_gen True`, the dataset mixture must include at least one generation-type dataset (interleaved editing or text-to-image) so that every packed batch carries VAE latents. This mirrors the training recipes used in the paper. For understanding-only fine-tuning (pure conversation SFT), set `--visual_gen False`.
