> The finished slide deck is the online Slides deck "ITRI626 Component Classification" (download as PowerPoint or PDF from its Share › Export menu). This outline is kept for reference; `oral_prep.md` has likely oral questions with answers.

# ITRI626 Oral Presentation Outline
## Deep Learning for Microcontroller and Electronic Component Classification

**Group**: Carlo Barnardo · Itumeleng Ntela · MEMBER 3  
**Speakers**: Slides 1–3 Itumeleng Ntela (problem, dataset) · Slides 4–6 Carlo (method, setup) · Slides 7–9 MEMBER 3 (results, errors) · Slide 10 all  
**Module**: ITRI626  
**Lecturer**: Prof. Absalom Ezugwu | **Facilitator**: Mr. Bernard Swanepoel  

---

### Slide 1: Title & Introduction
- **Title**: Comparative Classification of Microcontrollers & Electronic Components Using Deep Neural Networks
- **Domain**: Engineering & Manufacturing / Embedded Hardware / E-Waste Sorting
- **Aim**: Automated optical classification of MCU development boards and discrete components to support electronic recycling, lab inventorying, and assembly verification.

### Slide 2: Problem Motivation & Value
- **Industry Need**: Fast, non-destructive optical identification of electronic boards and components.
- **Challenges**:
  - Varied lighting, reflections on metallic solder joints, and component orientation.
  - Visual similarity between boards sharing form factors and solder mask colors (e.g., Nano vs. ESP8266, inductor vs. transformer).
- **Target Value**: Scalable automated visual inspection pipeline with minimal latency.

### Slide 3: Dataset Selection, Classes & Leakage Prevention
- **10 Target Classes**:
  - Boards: Arduino Nano, ESP8266, Heltec ESP32 LoRa, Raspberry Pi 3 (Tanner dataset, MIT)
  - Components: IC, Capacitor, Diode, Resistor, Inductor, Transformer (Faraji / Digi-Key)
- **Data Cleaning**: corrupt files, MD5 exact duplicates, perceptual-hash near-duplicates (cross-class ones removed as uncertain labels). Use real counts from `cleaning_report.json`.
- **Strict Data Partitioning**:
  - Split: **70% Train / 15% Validation / 15% Held-out Test**
  - Group-aware / stratified splitting with fixed random seed `42` to prevent data leakage between related capture sequences.

### Slide 4: Data Augmentation & Preprocessing
- **Image Resolution**: Resized to $224 \times 224$, normalized using ImageNet channel statistics ($\mu, \sigma$).
- **Realistic Augmentations (Train set only)**:
  - Random rotations ($\pm 20^\circ$) reflecting arbitrary orientation on workbench.
  - Horizontal & vertical flips (components and boards exhibit geometric symmetry).
  - Color jitter (accommodates ambient shadow and illumination variations).
  - Random affine translations (simulates off-center optical camera framing).

### Slide 5: Model Architectures & Transfer Learning Strategy
- **Three Meaningfully Distinct Model Families**:
  1. **Baseline Custom CNN** (from scratch):
     - 4-stage convolutional extractor (32, 64, 128, 256 channels), BatchNorm, ReLU, MaxPool, AdaptiveAvgPool, Dropout.
  2. **ResNet-18** (Transfer Learning):
     - Residual skip connections avoiding vanishing gradients; pre-trained on ImageNet-1k.
  3. **ViT-B/16** (Vision Transformer, Transfer Learning):
     - 196 image patches of 16×16 → 12 self-attention layers; no convolutions; ImageNet-1k pretrained.
- **Transfer strategy**: full fine-tuning, small LR + 2-epoch warm-up, new dropout + linear head.

### Slide 6: Experimental Setup & Training Workflow
- **Framework**: PyTorch + NixOS devenv environment.
- **Loss**: Cross-Entropy Loss with inverse-frequency class weights (counters class imbalance).
- **Optimization**: AdamW; LR $10^{-3}$ (CNN), $3\times10^{-4}$ (ResNet-18), $5\times10^{-5}$ (ViT); warm-up + cosine; effective batch 32 for all (ViT 16×2 accumulation).
- **Regularization**: Dropout (0.3), weight decay ($10^{-4}$), Early Stopping (patience: 7 epochs).

### Slide 7: Experimental Results & Model Comparison
- **Metric Comparison Table**:
  - Test Accuracy, Macro-averaged F1, Weighted F1, Parameter Count, Inference Latency.
- **Loss & Validation Curves**:
  - FILL IN from `outputs/figures/validation_comparison.png` after the real run.

### Slide 8: Confusion Matrix & Key Classification Errors
- **Confusion Matrix Analysis**:
  - FILL IN: the largest off-diagonal cells (see `top_confusions` in `outputs/results/*_results.json`).
- **Qualitative Error Analysis**:
  - Inspected top confident misclassifications.
  - FILL IN: show `misclassified_<best>.png` and name the causes you actually see.

### Slide 9: Limitations & Future Enhancements
- **Current Limitations**:
  - Single-board / single-component optical classification per frame.
- **Future Directions**:
  - Upstream object detection (YOLOv8/v11) to isolate multiple components on a single populated PCB.
  - Quantization (INT8) for microcontroller edge deployment via TensorRT / ONNX.

### Slide 10: Conclusion & Recommended Model
- FILL IN after the real run: best model by macro-F1, and the recommendation weighing accuracy against size/latency
  (ResNet-18 11.2 M params vs ViT-B/16 85.8 M).
