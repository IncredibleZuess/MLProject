# ITRI626 Oral Presentation Outline
## Deep Learning for Microcontroller and Electronic Component Classification

**Presenter / Group**: Carlo Barnardo  
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
  - Visual similarity between boards sharing form factors and solder mask colors (e.g., blue Nano vs. Blue Pill).
- **Target Value**: Scalable automated visual inspection pipeline with minimal latency.

### Slide 3: Dataset Selection, Classes & Leakage Prevention
- **10 Target Classes**:
  - MCUs: Arduino Uno, Arduino Nano, ESP32, ESP8266, Raspberry Pi Pico, STM32 Blue Pill
  - Components: Microcontroller IC, Resistor, Capacitor, LED
- **Data Cleaning**: Removed duplicates, corrupt files, and ambiguous samples.
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
  3. **MobileNetV3-Large** (Modern Architecture):
     - Inverted bottleneck blocks, depthwise separable convolutions, and Squeeze-and-Excitation attention for mobile/edge efficiency.

### Slide 6: Experimental Setup & Training Workflow
- **Framework**: PyTorch + NixOS devenv environment.
- **Loss**: Cross-Entropy Loss with inverse-frequency class weights (counters class imbalance).
- **Optimization**: AdamW ($\eta = 10^{-3}$ for CNN, $3 \times 10^{-4}$ for Transfer), Cosine Annealing scheduler.
- **Regularization**: Dropout (0.3), weight decay ($10^{-4}$), Early Stopping (patience: 5 epochs).

### Slide 7: Experimental Results & Model Comparison
- **Metric Comparison Table**:
  - Test Accuracy, Macro-averaged F1, Weighted F1, Parameter Count, Inference Latency.
- **Loss & Validation Curves**:
  - Custom CNN shows slower convergence and higher final loss.
  - ResNet-18 and MobileNetV3 converge rapidly with stable validation curves.

### Slide 8: Confusion Matrix & Key Classification Errors
- **Confusion Matrix Analysis**:
  - Strong discrimination across board types and components.
  - Primary confusion points: Arduino Nano vs. STM32 Blue Pill (both rectangular blue PCBs).
- **Qualitative Error Analysis**:
  - Inspected top confident misclassifications.
  - Specular reflection on shiny IC packaging obscured chip markings.

### Slide 9: Limitations & Future Enhancements
- **Current Limitations**:
  - Single-board / single-component optical classification per frame.
- **Future Directions**:
  - Upstream object detection (YOLOv8/v11) to isolate multiple components on a single populated PCB.
  - Quantization (INT8) for microcontroller edge deployment via TensorRT / ONNX.

### Slide 10: Conclusion & Recommended Model
- Pre-trained models substantially outperform the scratch-trained custom CNN baseline.
- **Final Recommendation**: **MobileNetV3-Large**
  - Yields competitive accuracy/F1 compared to ResNet-18 while cutting parameter count by $>60\%$ and delivering the fastest inference latency, making it ideal for edge sorting devices.
