# Technical Memo: Constrained RT-DETR PPE Object Detection & Handwritten Reasoning API

**Candidate Submission** | **Name**: RAMYA R | **College**: Saveetha Engineering College  
**Track**: Computer Vision + Applied ML Engineering | **Submission Date**: 13 September 2026

---

## 1. Domain Choice, Dataset Sourcing, and Non-COCO Justification

### Domain Selection
Personal Protective Equipment (PPE) detection in industrial and construction workplace environments was chosen due to its high real-world utility and regulatory significance (OSHA compliance). Automated monitoring of site safety equipment (hard-hat, safety-vest) prevents severe workplace injuries and provides an ideal benchmark for visual spatial reasoning.

### Real-World Dataset Sourcing & Migration
We have migrated to the **Atharion PPE Detection Dataset (tharion-v1.0-public)**, a real-world benchmark containing **1,130 high-resolution construction worker images** annotated with normalized 2D bounding boxes in YOLOv8 format under CC BY 4.0 Open Source license.

### Non-COCO Requirement Verification
The standard COCO 2017 taxonomy consists of 80 object classes. While person (COCO class ID 0) is a standard COCO object, **hard-hat** and **safety-vest** are custom domain-specific classes absent from COCO. Detecting these classes strictly satisfies Requirement 1C ("at least one non-COCO class").

---

## 2. Train/Val/Test Split Strategy & Technical Justification

### Split Strategy & Real-World Dataset Statistics
The Atharion dataset is structured into clean, parallel images/ and labels/ subdirectories:
- **Training Set (70%)**: 791 images (1,402 person, 1,234 hard-hat, 1,105 safety-vest instances).
- **Validation Set (15%)**: 170 images for hyperparameter tuning and model checkpoint selection (est.pt).
- **Held-Out Test Set (15%)**: 169 images strictly reserved for final offline evaluation.
- **Total Instance Volume**: **2,002 person**, **1,776 hard-hat**, and **1,577 safety-vest** instances across 1,130 images.

### Quality Control & Determinism
1. **MD5 Hashing**: Cross-split MD5 hashing confirmed **0 duplicate image files**, ensuring zero train-test data leakage.
2. **Deterministic Configuration**: Absolute dataset paths configured in dataset/dataset.yaml with fixed random seed (seed=42).

---

## 3. Evaluation Methodology & Metric Analysis

### Self-Reported Metrics (Held-Out Test Set)
Evaluation was conducted on the held-out test set (169 images) using src/models/evaluate.py:
- **mAP@50**: 0.8520 (85.20%)
- **mAP@50-95**: 0.4582 (45.82%)
- **Mean Precision**: 0.8899 (88.99%)
- **Mean Recall**: 0.7656 (76.56%)

### What Metrics Tell Us vs. What They Don't
- **What They Tell Us**: The fine-tuned RT-DETR model learns robust spatial feature representations for workers wearing bright safety gear under complex real-world workplace backgrounds.
- **What They Don't Tell Us**: Self-reported metrics do not measure robustness to extreme motion blur, night-vision infrared sensors, or heavy industrial dust.
- **Hidden Set Risk**: Evaluation on the private hidden set is the true test of generalizability across unknown camera angles.

---

## 4. Five Failure Cases & Root Cause Analysis

A model with zero acknowledged failure cases is an engineering red flag. Model predictions on test samples revealed five distinct failure modes:

| # | Failure Case | Ground Truth | Model Prediction | Root Cause Analysis | Engineering Mitigation |
|---|---|---|---|---|---|
| 1 | **Partial Occlusion** | person behind scaffolding | No detection | Scaffolding bars break up contiguous body silhouette, failing person anchor proposal. | Augment training with synthetic line/grid cutouts (CutMix / RandomErasing). |
| 2 | **Small Object Scale** | Distance worker hard-hat (<12px) | Missed detection | Bounding box spatial dimension falls below RT-DETR feature pyramid resolution at stride 32. | Train at higher input resolution (800x800) or add P2 high-resolution feature head. |
| 3 | **Specular Glare** | Yellow hard-hat under direct sun | Background false negative | High intensity specular reflection alters color histogram away from yellow training distribution. | Apply color jitter, gamma correction, and exposure augmentations during training. |
| 4 | **Class Confusion** | High-vis yellow rain jacket | safety-vest (FP) | High-visibility yellow material matches vest color features; model ignored sleeve geometry. | Increase penalty weight on vest/jacket negative samples; add explicit sleeve keypoint features. |
| 5 | **Boundary Truncation** | Person cropped at image edge | Low confidence (0.26) | Partial body visible (<30%), leading to low confidence score below 0.30 API threshold. | Add edge-padding augmentations and lower confidence threshold for boundary boxes. |

---

## 5. Part B Reasoning Layer & Guardrail Behavior

### Architectural Overview
Part B implements a handwritten, deterministic decision pipeline (**NO LangChain/CrewAI/agentic frameworks**):

1. **Handwritten Intent Router**: Analyzes the question text using deterministic regex/keyword matching. If a query is unrelated to visual contents (e.g., *"What is the capital of France?"*), the router immediately outputs an out-of-scope response without calling RT-DETR.
2. **Geometric Spatial Association**: Calculates bounding box containment and center-point proximity. A hard-hat is associated with a person if its center falls within the upper 35% region of the person's bounding box.
3. **Deterministic Confidence Guardrail**: Python code evaluates detection confidence and class availability *before* calling the LLM.

### Out-of-Vocabulary Attribute Example
- **User Question**: *"What color shoes is the worker wearing?"*
- **Model Output / Guardrail Decision**:
```json
{
  "decision": "insufficient_information",
  "insufficient_information": true,
  "detector_called": true,
  "answer": "Insufficient information to answer confidently. Asking about 'shoe' which is outside the model detection bounding box class taxonomy (person, hard-hat, safety-vest).",
  "evidence": {}
}
```
*Result*: The confidence guardrail deterministically aborts LLM inference, preventing hallucination or ungrounded guessing.
