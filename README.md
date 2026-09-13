# Constrained Object Detection & Reasoning API (RT-DETR)

An end-to-end Computer Vision & Applied ML system fine-tuned on the real-world **Atharion PPE Detection Dataset (`atharion-v1.0-public`)** using **RT-DETR** (`rtdetr-l.pt`), featuring a handwritten, deterministic reasoning layer and FastAPI endpoints.

---

## Architecture & Workflow

```text
                        [ User Question ]
                                |
                    [ Handwritten Intent Router ]
                                |
           +--------------------+--------------------+
           |                                         |
[ Unrelated / Out-of-Scope ]               [ Image-Related Query ]
           |                                         |
  Returns Non-Image Response                 [ RT-DETR Inference ]
    (RT-DETR NOT Called)                             |
                                           [ Structured Detections ]
                                                     |
                                           [ Geometric Association Logic ]
                                              (IoU / Box Containment)
                                                     |
                                           [ Deterministic Guardrail ]
                                                     |
                           +-------------------------+-------------------------+
                           |                                                   |
               [ Insufficient Evidence ]                             [ Sufficient Evidence ]
                           |                                                   |
               Returns "Insufficient info..."                        [ Google Gemini LLM ]
                   (Gemini NOT Called)                                         |
                                                                     Plain-Language Answer

## Real-World Dataset Statistics & Compliance

- **Dataset**: Atharion PPE Detection Dataset (tharion-v1.0-public) under CC BY 4.0 License.
- **Total Images**: **1,130 real-world workplace images**.
- **Split Distribution**:
  - **Train Set (70%)**: 791 images (1,402 person, 1,234 hard-hat, 1,105 safety-vest instances).
  - **Val Set (15%)**: 170 images for checkpoint selection (est.pt).
  - **Held-Out Test Set (15%)**: 169 images strictly reserved for final offline evaluation.
- **Non-COCO Verification**: Detects hard-hat and safety-vest (both absent from standard COCO 80-class taxonomy), satisfying Requirement 1C.
- **Quality Control**: 0 corrupt images, 0 missing labels, 0 cross-split MD5 duplicate hashes.

---

## Key Compliance Highlights

1. **No Agentic Frameworks**: Built entirely without LangChain, LangGraph, CrewAI, or AutoGen. Part B is a single handwritten Python decision engine.
2. **No AutoML**: Full custom PyTorch training & validation scripts using Ultralytics RT-DETR (
tdetr-l.pt).
3. **Reproducibility**: Deterministic data splitting with seed=42, strict requirements pinning, logged hyperparameters, and explicit execution steps.
4. **Deterministic Guardrails**: LLM cannot guess or hallucinate detections; guardrail returns explicit "insufficient information" when evidence is ambiguous or low-confidence.
5. **Bonus Included**: Full Docker containerization (Dockerfile, docker-compose.yml), visual rendering endpoint (POST /detect/visualize), structured Python logging, and FastAPI exception handlers.

---

## Quickstart & Installation

### 1. Environment Setup
`ash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
`

### 2. Dataset Verification (Step 0)
`ash
python src/data/verify_dataset.py
`

### 3. Model Training (Part A)
`ash
python src/models/train.py --data dataset/dataset.yaml --epochs 10 --batch 8 --imgsz 640 --device cpu --seed 42
`

### 4. Model Evaluation on Held-Out Test Set
`ash
python src/models/evaluate.py --weights runs/detect/runs/rtdetr_ppe_final/weights/best.pt --data dataset/dataset.yaml --split test
`

### 5. Run All 15 Unit & API Tests
`ash
python -m pytest tests/test_reasoning.py tests/test_api.py
`

### 6. Launch FastAPI Backend
`ash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
`

---

## Docker Containerization (Bonus 10%)

### Build & Run Docker Container:
`ash
docker build -t rt-detr-ppe .
docker run -p 8000:8000 rt-detr-ppe
`

### Or using Docker Compose:
`ash
docker-compose up --build
`

---

## API Documentation & Example Payloads

### Endpoint 1: POST /detect
Accepts an image and returns detected bounding boxes, classes, and confidences.

**Example Request:**
`ash
curl -X POST "http://localhost:8000/detect" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@test_image.jpg"
`

**Example Response:**
`json
{
  "success": true,
  "num_detections": 2,
  "summary": {
    "person": 2,
    "hard-hat": 2,
    "safety-vest": 2
  },
  "detections": [
    {
      "cls_id": 0,
      "class": "person",
      "confidence": 0.9632,
      "bbox": [0.3191, 0.2614, 0.5958, 0.8579]
    },
    {
      "cls_id": 1,
      "class": "hard-hat",
      "confidence": 0.8504,
      "bbox": [0.0705, 0.3360, 0.1820, 0.4052]
    }
  ],
  "inference_time_ms": 42.5
}
`

### Endpoint 2: POST /detect/visualize (Bonus Endpoint)
Accepts an image and returns a rendered PNG image showing 2D bounding boxes and class labels.

### Endpoint 3: POST /reason
Accepts an image and a natural-language question. Performs handwritten intent routing, spatial relationship analysis, deterministic guardrails, and structured LLM reasoning.

**Example Request 1 (Compliant Query):**
`ash
curl -X POST "http://localhost:8000/reason" -F "question=Is everyone in this image wearing a hard hat and safety vest?" -F "file=@test_image.jpg"
`

**Example Response 1:**
`json
{
  "success": true,
  "decision": "structured_reasoning_success",
  "insufficient_information": false,
  "detector_called": true,
  "answer": "Yes, 2 worker(s) detected in the image and all are wearing hard hats and safety vests.",
  "evidence": {
    "spatial_relationships": {
      "total_persons": 2,
      "total_hard_hats": 2,
      "total_safety_vests": 2,
      "person_compliance": [
        {
          "person_id": 0,
          "has_hard_hat": true,
          "has_safety_vest": true
        }
      ]
    }
  }
}
`

**Example Request 2 (Out-of-Scope Query - RT-DETR Skipped):**
`ash
curl -X POST "http://localhost:8000/reason" -F "question=What is the capital of France?" -F "file=@test_image.jpg"
`

**Example Response 2:**
`json
{
  "success": true,
  "decision": "unrelated_query",
  "insufficient_information": false,
  "detector_called": false,
  "answer": "This question is unrelated to the visual contents of the image.",
  "evidence": {}
}
`

**Example Request 3 (Insufficient Information Trigger):**
`ash
curl -X POST "http://localhost:8000/reason" -F "question=What color shoes is the worker wearing?" -F "file=@test_image.jpg"
`

**Example Response 3:**
`json
{
  "success": true,
  "decision": "insufficient_information",
  "insufficient_information": true,
  "detector_called": true,
  "answer": "Insufficient information to answer confidently. Asking about 'shoe' which is outside the model detection bounding box class taxonomy (person, hard-hat, safety-vest).",
  "evidence": {}
}
`
