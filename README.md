# ITRI626: Microcontroller & Electronic Component Classification

**Group:** Carlo Barnardo (43870449) · Itumeleng Ntela (43528309) · MEMBER 3 (STUDENT NO.)
**Lecturer:** Prof. Absalom Ezugwu · **Facilitator:** Mr. Bernard Swanepoel

Ten-class image classification (4 development boards + 6 component types), comparing three
model families on one leakage-free split:

| Family | Model | Weights |
|---|---|---|
| Plain CNN (baseline) | 4-stage custom CNN, 0.42 M params | random init |
| Residual CNN | ResNet-18, 11.2 M params | ImageNet-1k (torchvision) |
| Vision Transformer | ViT-B/16, 85.8 M params | ImageNet-1k (torchvision) |

MobileNetV3-Large is still available as an optional extra (`--model mobilenet_v3`), but it is
not one of the three main models: the brief lists ResNet and MobileNetV3 in the same group.

## Dataset

`data/raw/` (2,440 images) is built from two public GitHub datasets. A visual review excluded 509 wrong or unusable images
(`data/manual_exclude.csv`, applied automatically by `prepare`), leaving **1,902 images** after cleaning:

| Classes | Source | Licence |
|---|---|---|
| arduino_nano, esp8266, esp32_heltec_lora, raspberry_pi_3 (~45 each) | [TannerGilbert/Tensorflow-Object-Detection-API-Train-Model](https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model): bounding boxes cropped | MIT |
| ic, capacitor, diode, resistor, inductor, transformer (≤400 each) | [pouria-faraji/Electronic-Components-Classification](https://github.com/pouria-faraji/Electronic-Components-Classification) (Digi-Key + Kaggle `aryaminus/electronic-components`) | no licence file: acknowledge, and check with the lecturer |

To rebuild it from scratch:
```bash
git clone --depth 1 https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model.git ext/boards
git clone --depth 1 https://github.com/pouria-faraji/Electronic-Components-Classification.git ext/components
python -m mlproject.data.build_public_dataset --boards ext/boards --components ext/components
```
To add your own photos, put them in `data/raw/<class>/`, give photos of one physical object a shared prefix
(`board03__01.jpg`, `board03__02.jpg`) and see `data/DATASET.md`.

## Setup

**Windows / macOS / Linux (pip):**
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
```
**uv:** `uv sync`, then prefix commands with `uv run`.
**NixOS:** `devenv shell`.

Check the GPU: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"`

## Reproducing every result

```bash
python -m mlproject prepare          # clean data/raw, write data/splits/*.csv, classes.json, cleaning_report.json
python -m mlproject train --model custom_cnn
python -m mlproject train --model resnet18
python -m mlproject train --model vit_b_16
python -m mlproject evaluate --model custom_cnn
python -m mlproject evaluate --model resnet18
python -m mlproject evaluate --model vit_b_16
python -m mlproject compare          # comparison, validation-curve and per-class F1 figures
python -m mlproject report-assets    # LaTeX tables + figures for report/report.tex
```
or all at once: `python -m mlproject run-all`.
Then run `notebooks/microcontroller_classification.ipynb` top to bottom (it reuses the saved checkpoints)
and compile the report: `cd report && latexmk -pdf report.tex`.

Hyperparameters live in one place, `ProjectConfig.model_settings` in `src/mlproject/config.py`.
Seed: 42. Every training run logs its settings, library versions and GPU to
`outputs/models/<model>/history.json`.

**GPU memory:** ViT-B/16 uses batch 16 with 2 gradient-accumulation steps (effective batch 32) and
mixed precision. If you still run out of memory, use `--batch-size 8 --accum-steps 4`.

**Quick smoke test:** `python -m mlproject run-all --quick 2` trains every model for 2 epochs only.

## Outputs

| Path | Contents |
|---|---|
| `data/splits/` | train/val/test CSVs, `classes.json` (counts, seed, split method), `cleaning_report.json` (every removed file and why) |
| `outputs/models/<model>/` | `best_model.pt`, `history.json` (curves, hyperparameters, environment) |
| `outputs/results/` | `<model>_results.json` (all metrics, confusion matrix), `<model>_test_predictions.csv` |
| `outputs/figures/` | class distribution & examples, learning curves, confusion matrices, comparison, per-class F1, correct / misclassified grids |
| `report/generated/` | auto-generated LaTeX tables used by `report.tex` |

## Code map

```
src/mlproject/
  config.py          paths, classes, per-model hyperparameters
  split.py           cleaning (corrupt, MD5 duplicates, dHash near-duplicates) + group-aware split + leakage check
  data/download_and_prep.py   prepare pipeline, synthetic demo generator
  dataset.py         transforms (augmentation on train only), DataLoaders, class weights
  models/            custom_cnn.py, resnet_model.py, transformer_model.py (ViT / MobileNetV3), factory.py
  train.py           AdamW, warm-up + cosine, grad accumulation, AMP, early stopping, logging
  evaluate.py        test metrics, confusion matrix, latency, error lists
  visualize.py       all figures
  report_assets.py   results -> LaTeX tables
  main.py            CLI
```
