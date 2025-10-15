# Model Card for UniMedVL

UniMedVL is the first unified medical foundation model for seamless multimodal understanding and generation, following a clinically-inspired Observation-Knowledge-Analysis framework.

## Model Details

### Model Description

UniMedVL unifies medical multimodal understanding and generation within a single 14B-parameter architecture. The model supports visual question answering, medical report generation, text-to-medical-image synthesis, cross-modal translation, and virtual staining across 9 imaging modalities (CXR, CT, MRI, Ultrasound, Histopathology, Retinal Fundus, OCT, Endoscopy, and Dermoscopy).

- **Developed by:** [Anonymous - Under Review]
- **Model type:** Unified Vision-Language Model
- **Language(s):** English (medical domain)
- **License:** Apache License 2.0
- **Model Size:** 14B parameters

### Model Sources

- **Repository:** https://github.com/[repository_name]
- **Project Page:** https://uni-medical.github.io/UniMedVL_Web/
- **Paper:** https://arxiv.org/abs/xxxxx (Coming Soon)

## Uses

### Direct Use

The model can be directly used for:
- **Medical Visual Question Answering**: Answer clinical questions about medical images
- **Medical Report Generation**: Generate radiology reports from medical images
- **Text-to-Medical-Image Synthesis**: Generate medical images from textual descriptions
- **Cross-Modal Translation**: Convert between different medical imaging modalities
- **Virtual Staining**: Transform H&E images to IHC staining

### Out-of-Scope Use

- **Clinical Decision Making**: This model is for research purposes only and should NOT be used for actual clinical diagnosis or treatment decisions
- **Patient Care**: Not approved for direct patient care applications
- **Commercial Use**: Currently restricted to academic research

## Training Details

### Training Data

The model is trained on **UniMed-5M**, a comprehensive medical multimodal dataset containing:
- 5.6M+ high-quality image-text pairs
- 9 medical imaging modalities
- Three-stage quality verification and expert validation
- Tasks: Understanding (I2T), Generation (T2I), and Interleaved multimodal tasks

### Training Procedure

**Three-Stage Progressive Curriculum Learning:**

1. **Stage 1 - Foundation Training** (85K steps)
   - Visual-language alignment
   - Data ratio: 75% I2T, 25% T2I

2. **Stage 2 - Instruction Tuning** (120K steps)
   - Cross-modal understanding enhancement
   - Data ratio: 40% I2T, 45% T2I, 10% Interleaved

3. **Stage 3 - Unified Training** (70K steps)
   - Advanced multimodal synthesis
   - Data ratio: 37% I2T, 35% T2I, 25% Interleaved

#### Training Hyperparameters

- **Training regime:** Mixed precision (bf16)
- **Total training steps:** 275K
- **Base models:** Qwen2-VL (vision encoder), Llama-3 (language model)

## Evaluation

### Medical Visual Question Answering

| Dataset | UniMedVL (14B) | GMAI-VL (7B) | HealthGPT-L14 (14B) |
|---------|----------------|--------------|---------------------|
| VQA-RAD | **61.9** | 66.3 | 58.3 |
| SLAKE | **75.4** | 72.9 | 64.5 |
| PathVQA | **53.5** | 39.8 | 44.4 |
| OmniMedVQA | **85.8** | 88.5 | 74.4 |

### Medical Image Generation

Average performance across 8 modalities:
- **gFID**: 96.29 (vs. Bagel 215.49)
- **BioMedCLIP Score**: 0.706 (vs. Bagel 0.660)

### Interleaved Multimodal Tasks

| Task | Metric | UniMedVL | Specialized SOTA |
|------|--------|----------|------------------|
| Virtual Staining (H&E→IHC) | PSNR/SSIM | 20.27/0.456 | 21.16/0.477 |
| MRI Super-Resolution (4×) | PSNR/SSIM | 27.29/0.890 | 31.99/0.939 |
| Cross-Modal Synthesis (MRI) | PSNR/SSIM | 25.07/0.882 | 25.38/0.889 |

## Bias, Risks, and Limitations

- **Research Only**: Not validated for clinical use
- **Dataset Bias**: Performance may vary across different populations and imaging protocols
- **Hallucination**: May generate plausible but incorrect medical information
- **Modality Coverage**: Limited to 9 imaging modalities included in training data

### Recommendations

Users should:
- Validate all outputs with qualified medical professionals
- Be aware of potential biases in training data
- Not use for clinical decision-making without proper validation
- Report any concerning model behaviors

## How to Get Started with the Model

```python
# Coming soon - Model weights and inference code will be released upon paper acceptance
# Example usage will be provided with model release
```

## Citation

**BibTeX:**

```bibtex
@article{unimedvl2024,
  title={UniMedVL: Unifying Medical Multimodal Understanding and Generation through Observation-Knowledge-Analysis},
  author={[Anonymous Authors]},
  journal={arXiv preprint arXiv:xxxxx},
  year={2024}
}
```

## Model Card Contact

For questions and feedback, please open an issue on the GitHub repository or visit the project page.

---

**Note**: This project is currently under anonymous review. Code, model weights, and dataset will be released upon paper acceptance.
