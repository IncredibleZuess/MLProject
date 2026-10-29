# ITRI626 oral: likely questions and short answers

Both of you must be able to answer anything. Learn the **key numbers** first, then practise saying each answer out loud in your own words.

## Key numbers

| | |
|---|---|
| Raw images → after review and cleaning | 2,440 → **1,902** (509 removed by visual review, 10 exact duplicates, 19 cross-class near-duplicates) |
| Split (seed 42) | **1,414 train / 264 val / 224 test**, about 70/15/15, group-aware |
| Classes | 10: arduino_nano, esp8266, esp32_heltec_lora, raspberry_pi_3, ic, capacitor, diode, resistor, inductor, transformer |
| Imbalance | resistor 382 vs. Nano / Pi 3 with 43 each (about 9 : 1) |
| Custom CNN | 50.9% accuracy, macro-F1 **0.543**, 0.42 M params, 1.1 ms/image |
| ResNet-18 | **90.2%**, macro-F1 **0.901**, 11.2 M params, 1.4 ms/image |
| ViT-B/16 | 88.0%, macro-F1 **0.898**, 85.8 M params, 12.1 ms/image |
| Hardware | Google Colab, Tesla T4 (14.6 GB), PyTorch 2.11 |

## Problem and data

**Why this topic?**
Identifying boards and components automatically helps with e-waste sorting, lab inventory and checking assemblies. Several classes look alike, so it is a real fine-grained problem.

**Where does the data come from, and what are the licences?**
- Boards: Gilbert Tanner's microcontroller detection dataset on GitHub, MIT licence. We cropped each bounding box into its own image.
- Components: Faraji's GitHub dataset, made from Digi-Key catalogue images and a Kaggle dataset. It has no licence file, so we use it for coursework only and credit it. *(Say what the lecturer told you.)*

**Why did the classes change from the original plan (Uno, Pico, Blue Pill)?**
We could not find openly licensed images of those boards, and 20 images per class would not have been enough.

**How did you clean the data?**
1. Visual review of every image. 509 were removed where the labelled object was not the main subject, for example whole circuit boards filed as IC, or transformers filed as inductor. Every removal and its reason is in `data/manual_exclude.csv`.
2. Corrupt files.
3. Exact duplicates, found by MD5 hash.
4. Near-duplicates, found with a perceptual hash (dHash). If a near-duplicate pair has different labels, both images are removed. If they have the same label, they are put in one group.

**How did you prevent data leakage?**
- Photos of the same object, and near-duplicate images, share a group id.
- `StratifiedGroupKFold` puts each whole group into one split.
- The code checks that no group id or file hash appears in two splits.

**Is there still leakage risk?**
Yes, for the boards. Each board class shows one physical board, so test crops show the same board as training crops. That makes board scores optimistic.

**How did you handle class imbalance?**
- Cross-entropy loss weighted by inverse class frequency.
- Macro-F1 and balanced accuracy as the main metrics.

**Why these augmentations?**
- Flips and ±20° rotation: boards lie in any orientation on a bench.
- Colour jitter: lighting varies between rooms and cameras.
- Small shifts: objects are not always centred.

They are applied to the training set only. We did not use MixUp or CutMix, because blending two boards gives images that cannot exist.

## Models and setup

**Why these three models?**
They are three different families:
- a plain CNN trained from scratch (the baseline);
- a residual CNN with transfer learning (ResNet-18);
- a transformer with no convolutions (ViT-B/16).

We dropped MobileNetV3 because the brief lists it in the same group as ResNet.

**What is a residual connection?**
The block learns a correction to its input, y = F(x) + x. The shortcut lets gradients flow through, so very deep networks can be trained.

**How does ViT work?**
- The image is cut into 196 patches of 16×16 pixels, and each patch becomes a token.
- 12 layers of self-attention let every patch attend to every other patch.
- A class token is used for the final prediction.
- It has no built-in locality, so it relies on pretraining.

**What does "transfer learning" mean here?**
- Start from ImageNet-1k weights and replace the 1000-class head with a 10-class head.
- Fine-tune all layers with a small learning rate and a 2-epoch warm-up.
- The warm-up stops large early gradients from the new head damaging the pretrained features.

**Why different learning rates? Is the comparison fair?**
- Same split, same augmentations, same loss and optimiser, and the same effective batch of 32 for all models.
- The CNN starts from random weights, so it needs bigger steps (1e-3).
- Pretrained models need gentle updates: 3e-4 for ResNet-18, and 5e-5 for ViT, because ViTs are known to diverge at CNN learning rates.
- We changed nothing after seeing validation or test results.

**What is gradient accumulation and why for ViT?**
ViT runs batches of 16 and adds up the gradients of two batches before each update. That gives an effective batch of 32 while fitting in GPU memory.

**How does early stopping work?**
- Training stops if validation loss has not improved for 7 epochs.
- We keep the checkpoint with the lowest validation loss.
- The test set is used once, at the end.

## Results

**Which model is best?**
ResNet-18: 90.2% accuracy, macro-F1 0.901. ViT-B/16 is practically the same.

**Is ResNet-18 really better than ViT?**
Not meaningfully. On 224 test images one image is about 0.45 percentage points, so the 2-point gap is five images. The order also flips on the validation set (ViT 0.945 vs. ResNet 0.933 val macro-F1). We would need several seeds to separate them.

**Why is the CNN so bad?**
It underfits: training accuracy is only 54% after 38 epochs, and training loss stays above 1.0. About 1,400 images are not enough to learn good visual features from scratch.

It also seems to use photo style as a shortcut: it predicts a board class for 17 component images. The pretrained models bring general features from 1.28 M ImageNet images.

**What does the confusion matrix show?**
- Errors are concentrated among the small two-lead parts: capacitor, diode and resistor.
- ResNet-18's most common error is IC → diode (3). ViT-B/16's is capacitor → diode (7).
- Our expected confusions were rare: Nano vs. ESP8266 happened once, and inductor vs. transformer never for ResNet-18.

**Show an error and explain it.**
- A diode in a TO-220 case, which looks like a transistor, predicted as a resistor.
- A chip on a crowded board predicted as ESP8266: the model reacted to the board context.
- An Arduino Nano on a breadboard predicted as ESP8266: an unusual background, because the training Nanos are on plain surfaces.

**Anything surprising?**
Almost all errors have more than 90% confidence, so the raw confidence cannot be used to flag uncertain images without calibration.

**Which classes are easiest and hardest?**
- Easiest: the boards (Heltec ESP32 and Pi 3 reach F1 1.00, but these scores are optimistic), then transformer and resistor.
- Hardest: diode, capacitor and IC.

## Limitations and next steps

**What are the main limitations?**
1. Each board class shows one physical board.
2. Source confound: bench photos for boards, catalogue photos for components.
3. A small test set and a single seed.
4. Only four board types.
5. Catalogue labels are noisy: we removed 21% of the images.

**What would you do next?**
- Photograph several different boards per class, and add the Uno and Pico.
- Add bench photos of the components.
- Run several seeds and report confidence intervals.
- Calibrate the confidence scores.
- Use an object detector for boards with many components.

**Which model would you deploy?**
ResNet-18. It is as accurate as ViT-B/16 but 7.7× smaller and about 8× faster, and it trains in under 5 minutes on a free GPU.

## Process questions (answer honestly)

**Who did what?**
Use the group contributions section of the report.

**Did you use AI tools?**
Yes: Itumeleng used Claude (an AI assistant) for coding help, the dataset review and drafting. Say what the module's AI policy allows, and be ready to explain every part yourself. That is the point of this sheet.
