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

**Training Configuration:**
- Mixed precision training (bf16)
- Total training steps: 275K
- Base models: Qwen2-VL (vision encoder), Llama-3 (language model)

## Citation

If you find this model useful, please cite:

> UniMedVL: Unifying Medical Multimodal Understanding and Generation through Observation-Knowledge-Analysis. [Anonymous Authors]. arXiv preprint arXiv:xxxxx (2024).

## Contact

For questions and feedback, please open an issue on the GitHub repository or visit the project page.

---

**Note**: This project is currently under anonymous review. Code, model weights, and dataset will be released upon paper acceptance.
