# ITRI626: Deep Learning for Image Classification
## Microcontroller & Electronic Component Classification

This repository contains the complete practical project implementation for **ITRI626 (Deep Learning for Image Classification)** at North-West University.

The project classifies microcontroller development boards and electronic components from optical images using deep neural networks, comparing three distinct architectural families under a rigorous experimental methodology.

---

## Target Classes (10 Classes)
1. **`arduino_uno`**: ATmega328P standard development board
2. **`arduino_nano`**: Compact breadboard-friendly MCU board
3. **`esp32`**: Dual-core Tensilica Wi-Fi/Bluetooth SoC board
4. **`esp8266`**: Single-core Wi-Fi IoT development board (NodeMCU)
5. **`raspberry_pi_pico`**: RP2040 ARM Cortex-M0+ dual-core board
6. **`stm32_bluepill`**: ARM Cortex-M3 32-bit development board
7. **`microcontroller_ic`**: Discrete IC package / microcontroller chip
8. **`resistor`**: Axial/SMD resistor components
9. **`capacitor`**: Electrolytic canister & ceramic capacitor
10. **`diode_led`**: Light-emitting diode & signal diode

---

## Model Families Implemented
In strict compliance with **ITRI626 Section 6**:
1. **Family 1 (Baseline)**: `CustomBaselineCNN`
   - 4-stage convolutional feature extractor (32, 64, 128, 256 filters) with BatchNorm, ReLU, MaxPool, AdaptiveAvgPool, and Dropout.
   - Trained from scratch (no pre-trained weights).
2. **Family 2 (Residual / Transfer Learning)**: `ResNetTransferModel` (ResNet-18)
   - Deep residual network pre-trained on ImageNet-1k with customized classification head.
3. **Family 3 (Modern / Inverted Residuals)**: `MobileNetOrViTModel` (MobileNetV3-Large)
   - Depthwise separable convolutions, inverted bottleneck blocks, and squeeze-and-excitation attention.

---

## Repository Structure
```text
MLProject/
├── devenv.nix                     # NixOS reproducible environment specification
├── devenv.yaml                    # devenv lock & inputs config
├── pyproject.toml                 # Project metadata & Python dependencies
├── README.md                      # Project documentation and reproduction steps
├── ITRI626_Project_New-1.pdf       # ITRI626 assignment & rubric brief
├── data/
│   ├── raw/                       # Raw class image directories (e.g. data/raw/arduino_uno/...)
│   ├── processed/                 # Processed & cleaned images
│   └── splits/                    # Reproducible split manifests (train.csv, val.csv, test.csv, classes.json)
├── src/
│   └── mlproject/
│       ├── __init__.py
│       ├── main.py                # Unified CLI entrypoint
│       ├── config.py              # Configuration dataclass (seed, paths, hyperparameters)
│       ├── dataset.py             # PyTorch Dataset, transforms, and DataLoaders
│       ├── split.py               # Data leakage prevention, cleaning, and stratified splitting
│       ├── train.py               # Training loop with early stopping, schedulers, checkpointing
│       ├── evaluate.py            # Test evaluation, per-class metrics, confusion matrix, error analysis
│       ├── visualize.py           # Plotting learning curves, comparisons, and error inspection
│       ├── models/
│       │   ├── __init__.py
│       │   ├── custom_cnn.py      # Baseline CNN architecture (Family 1)
│       │   ├── resnet_model.py    # ResNet-18 transfer learning architecture (Family 2)
│       │   ├── transformer_model.py # MobileNetV3 / ViT architecture (Family 3)
│       │   └── factory.py         # Model instantiation & parameter inspection
│       └── data/
│           ├── __init__.py
│           └── download_and_prep.py # Data ingestion & demo dataset generator
├── notebooks/
│   └── microcontroller_classification.ipynb # Interactive notebook with outputs & plots
├── report/
│   └── report.tex                 # LaTeX project report template (10-15 pages)
├── presentation/
│   └── slides_outline.md          # 10-slide oral presentation outline
└── outputs/
    ├── models/                    # Saved best model checkpoints (*.pt) & history.json
    ├── figures/                   # Confusion matrices, learning curves, comparison bar charts
    └── results/                   # JSON evaluation metrics and per-class reports
```

---

## Getting Started on NixOS with devenv

### 1. Enter the devenv shell
On NixOS, enter the isolated shell with all necessary libraries and Python 3.12:
```bash
devenv shell
```
To verify the GPU and environment:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## Reproducing Experiments

### Step 1: Prepare the Dataset & Splits
To generate the demo dataset (for immediate testing and verification) and create the reproducible 70/15/15 train/val/test splits:
```bash
python -m mlproject prepare --demo --samples 40
```
*Note: If you have your own real image dataset, place images in `data/raw/<class_name>/*.jpg` and run `python -m mlproject prepare`.*

### Step 2: Train the 3 Model Families
Train each model family with fixed random seed `42`:
```bash
# Family 1: Baseline Custom CNN (from scratch)
python -m mlproject train --model custom_cnn --epochs 15 --lr 0.001

# Family 2: ResNet-18 (Transfer Learning)
python -m mlproject train --model resnet18 --epochs 12 --lr 0.0003

# Family 3: MobileNetV3-Large (Modern Architecture)
python -m mlproject train --model mobilenet_v3 --epochs 12 --lr 0.0003
```
Checkpoints and training histories (`history.json`) are automatically saved to `outputs/models/<model_name>/best_model.pt`.

### Step 3: Evaluate on Held-Out Test Set
Evaluate each model on the held-out test split:
```bash
python -m mlproject evaluate --model custom_cnn
python -m mlproject evaluate --model resnet18
python -m mlproject evaluate --model mobilenet_v3
```
This generates:
- Per-class precision, recall, and F1 scores
- Macro-averaged F1 and balanced accuracy
- Normalized confusion matrix heatmaps in `outputs/figures/confusion_matrix_<model>.png`
- Misclassified error analysis in `outputs/figures/misclassified_<model>.png`

### Step 4: Generate Multi-Model Comparison Chart
Generate comparative bar charts across all 3 models:
```bash
python -m mlproject compare
```
Outputs `outputs/figures/model_comparison.png` comparing Test Accuracy, Macro F1, Parameters (M), and Inference Latency (ms).

---

## ⚡ Automated Benchmark (Run Everything in One Shot)
You can run the entire data prep, training of all 3 models, evaluation, and chart generation with a single command:
```bash
python -m mlproject run-all
```

---

## Running on Windows (For Group Members)

Group members on Windows do **not** need Nix or devenv. You can run the code natively using standard Python or `uv`.

### Option A: Using standard Python (`venv` + `pip`)
1. Clone the repository and navigate into it:
   ```cmd
   git clone <repo-url>
   cd MLProject
   ```
2. Create and activate a Python virtual environment (Python 3.10, 3.11, or 3.12 recommended):
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   pip install -e .
   ```
4. Run the code or Jupyter:
   ```cmd
   python -m mlproject prepare --demo --samples 40
   python -m mlproject run-all
   jupyter lab
   ```

### Option B: Using `uv` on Windows (Fastest)
If group members have `uv` installed (`winget install astral-sh.uv`):
```cmd
uv sync
uv run python -m mlproject run-all
uv run jupyter lab
```

### Option C: Using Windows Subsystem for Linux (WSL2)
Teammates using WSL2 can install `devenv` and use the identical `devenv shell` workflow as on Linux.

---

## Running the Jupyter Notebook
Launch Jupyter inside the devenv shell:
```bash
devenv shell -- jupyter notebook notebooks/microcontroller_classification.ipynb
```
or inside `devenv shell`:
```bash
jupyter lab
```

---

## Rubric Compliance Summary
| ITRI626 Criterion | Requirement | Repository Implementation |
|---|---|---|
| **Problem Definition** | Clear scope and target classes | 10 classes (6 MCUs + 4 components) defined in `config.py` |
| **Dataset & Preparation** | Cleaning, distribution, leakage prevention | `split.py` & `download_and_prep.py` with 70/15/15 stratified split |
| **Model Selection** | 3 different model families, transfer learning | Custom CNN (scratch), ResNet-18 (transfer), MobileNetV3 (transfer) in `models/` |
| **Reproducibility** | Seed, hardware, hyperparams reported | Seed 42, AdamW, cosine annealing, metrics logged to JSON in `train.py` |
| **Results & Evaluation** | Test acc, macro F1, confusion matrix, loss curves | `evaluate.py` & `visualize.py` |
| **Error Analysis** | Qualitative inspection of misclassified samples | `plot_misclassified_samples` in `visualize.py` |
| **Report & Presentation** | LaTeX report + slides outline | `report/report.tex` + `presentation/slides_outline.md` |
