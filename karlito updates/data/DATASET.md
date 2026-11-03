# Dataset: sources, collection guide and record

`data/raw/` now holds a **real dataset** built from two public GitHub sources by
`python -m mlproject.data.build_public_dataset` (see the README). `SOURCES.json` in `data/raw/`
records the sources, the cap and the counts.

| Class | Source | Licence | Images |
|---|---|---|---|
| arduino_nano | Tanner (crops of bounding boxes) | MIT | 43 |
| esp8266 | Tanner | MIT | 47 |
| esp32_heltec_lora | Tanner | MIT | 44 |
| raspberry_pi_3 | Tanner | MIT | 43 |
| ic | Faraji (Digi-Key / Kaggle), random 400 of 1,503 | none stated | 400 |
| capacitor | Faraji, 400 of 535 | none stated | 400 |
| diode | Faraji, 400 of 518 | none stated | 400 |
| resistor | Faraji, 400 of 470 | none stated | 400 |
| inductor | Faraji, all | none stated | 265 |
| transformer | Faraji, 400 of 747 | none stated | 400 |

**Manual review (done):** every image was inspected on numbered contact sheets (AI-assisted).
509 images are excluded via `data/manual_exclude.csv` (file + reason for each); `prepare` applies it.
After review and automatic cleaning: **1,902 images**. A group member should skim the CSV and confirm.

**Still to do by the group:**
1. Confirm the exclusion list (delete a row from the CSV to put an image back).
2. Ask the lecturer whether the unlicensed component images may be used with acknowledgement.
3. Optional but valuable: add your own photos of the board classes (they only have about 45 images each),
   or new boards such as the Arduino Uno. Add new class names to `target_classes` in `config.py`.

The rest of this file explains how to add images.

## 1. Folder layout

```
data/raw/
  arduino_nano/  esp8266/  esp32_heltec_lora/  raspberry_pi_3/
  ic/  capacitor/  diode/  resistor/  inductor/  transformer/
```
Folder names are the class labels. Any of .jpg .jpeg .png .bmp .webp works.

## 2. Group ids (prevents data leakage)

If several photos show the **same physical board or part**, give them the same group,
either with a double underscore in the file name or a sub-folder:

```
data/raw/arduino_nano/nano03__01.jpg   nano03__02.jpg   nano03__03.jpg  <- one physical board
data/raw/esp8266/nodemcu_b/0001.jpg     nodemcu_b/0002.jpg                 <- one physical board
```
Files without either convention are their own group. Near-identical copies (resized,
recompressed) are detected and grouped automatically.

## 3. Targets

- Aim for **at least 150 to 200 images per class** after cleaning (the more the better for the CNN baseline).
- Mix backgrounds, lighting, angles, both sides of the board, and clones (e.g. CH340 Uno clones).
- If you take your own photos: about 5 to 10 photos per physical object, then use the group naming above.
- Use only images whose licence allows academic use, and note every source below.

## 4. Possible public sources (check licence and class coverage before using)

- Kaggle and Roboflow Universe have several "electronic components" and "Arduino / ESP32" datasets.
  Roboflow ones are usually object-detection sets, so crop each box to one image per object.
- Mendeley Data "IoTKITs" (Arduino Uno/Nano, ESP32, Raspberry Pi…), DOI 10.17632/x5thzmkxhy.1.
- Your own photographs (the NWU electronics lab, your own kits). This is encouraged by the brief, and
  it lets you answer any question in the oral exam.

If a class can't reach a reasonable size, it is better to **drop or merge it** (update
`target_classes` in `src/mlproject/config.py` and the class table in the report) than to keep a class with 20 images.

## 5. Manual check log

| Class | Reviewed | Excluded | Kept (after auto-cleaning) | Main reasons |
|---|---|---|---|---|
| arduino_nano | 43 | 0 | 43 | none |
| esp8266 | 47 | 0 | 47 | none |
| esp32_heltec_lora | 44 | 0 | 44 | none |
| raspberry_pi_3 | 43 | 0 | 43 | none |
| ic | 400 | 218 | 182 | whole boards / computers / scenes, no single chip |
| capacitor | 400 | 43 | 355 | boards and scenes |
| diode | 400 | 102 | 291 | boards/breadboards; 16 images of ICs |
| resistor | 398 | 6 | 382 | film capacitors, jumper wire, packaging |
| inductor | 265 | 84 | 181 | 17 mains transformers, 32 historic induction coils, scenes |
| transformer | 400 | 56 | 334 | grid substations, UPS/rack equipment, scenes |

Reviewed by: Claude (AI assistant) on contact sheets · Confirmed by: MEMBER NAME

## 6. Extra boards considered and rejected
37 images from a collage of unknown origin (appears synthetic, no licence, one image per board, mostly classes we do not have) are in `data/extra_boards_not_used/`. They were not used in any experiment.
