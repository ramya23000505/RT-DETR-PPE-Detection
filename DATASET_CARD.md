# Atharion PPE Detection Dataset (v1.0-public)

## Current checked-in data

`dataset/` contains **1,130 real-world high-resolution construction worker images** with normalized 2D bounding boxes in YOLOv8 format under CC BY 4.0 Open Source license:
- **Training Set (70%)**: 791 images (1,402 person, 1,234 hard-hat, 1,105 safety-vest instances).
- **Validation Set (15%)**: 170 images for checkpoint selection.
- **Held-Out Test Set (15%)**: 169 images strictly reserved for final offline evaluation.

Classes are `person` (0), `hard-hat` (1), and `safety-vest` (2); the latter two are non-COCO custom classes.

## Quality Control & Verification

`src/data/verify_dataset.py` validates all labels, image dimensions, and cross-split MD5 hashes to guarantee 0 corrupt files, 0 missing annotations, and 0 cross-split image duplicates.

