import os
import io
import time
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from PIL import Image, ImageOps, ImageDraw

from ultralytics import RTDETR
from app.reasoning import handwritten_intent_router, process_reasoning_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rt_detr_ppe.api")

# Prevent Pillow decompression bomb attacks
Image.MAX_IMAGE_PIXELS = 25_000_000

app = FastAPI(
    title="Constrained RT-DETR PPE Object Detection & Reasoning API",
    description="FastAPI endpoints for RT-DETR PPE Object Detection and Handwritten Reasoning Layer.",
    version="1.0.0"
)

# Search candidates for trained best.pt weights ONLY (No silent COCO base fallback)
TRAINED_WEIGHTS_PATHS = [
    os.path.join("runs", "detect", "runs", "rtdetr_ppe_final", "weights", "best.pt"),
    os.path.join("runs", "rtdetr_ppe_final", "weights", "best.pt")
]

_model = None

def get_model():
    global _model
    if _model is not None:
        return _model
        
    target_path = None
    for path in TRAINED_WEIGHTS_PATHS:
        if os.path.exists(path):
            target_path = path
            break
            
    if not target_path:
        logger.error("Trained RT-DETR PPE model weights (best.pt) not found. Refusing fallback to COCO base model.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained RT-DETR model weights (best.pt) not found. Refusing fallback to pretrained COCO model."
        )
        
    logger.info(f"Loading trained RT-DETR PPE model from: {target_path}")
    try:
        _model = RTDETR(target_path)
        logger.info("RT-DETR PPE model loaded successfully.")
        return _model
    except Exception as e:
        logger.error(f"Failed to load trained RT-DETR model: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Model loading error: {str(e)}")

# Pydantic Schemas
class BoundingBox(BaseModel):
    cls_id: int
    class_name: str = Field(..., alias="class")
    confidence: float
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2] normalized bounding box coordinates")

class DetectResponse(BaseModel):
    success: bool
    num_detections: int
    summary: Dict[str, int]
    detections: List[BoundingBox]
    inference_time_ms: float

class ReasonResponse(BaseModel):
    success: bool
    decision: str
    insufficient_information: bool
    detector_called: bool
    answer: str
    evidence: Dict[str, Any]

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two normalized boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def apply_class_aware_nms(detections: List[dict], iou_threshold: float = 0.45) -> List[dict]:
    """
    Strict Class-Aware Non-Maximum Suppression (NMS).
    Eliminates duplicate overlapping bounding boxes for the same class with IoU > 0.45.
    """
    if not detections:
        return []
        
    # Sort detections by confidence descending
    sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    kept_dets = []
    
    for current in sorted_dets:
        discard = False
        for kept in kept_dets:
            if current["cls_id"] == kept["cls_id"]:
                iou = compute_iou(current["bbox"], kept["bbox"])
                if iou > iou_threshold:
                    discard = True
                    break
        if not discard:
            kept_dets.append(current)
            
    return kept_dets

def validate_and_load_image(file_bytes: bytes, filename: str = "") -> Image.Image:
    if not file_bytes or len(file_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded image file is empty.")
    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File size exceeds maximum allowed 15MB limit.")
        
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image format or corrupt file: {str(e)}")
        
    w, h = img.size
    if w > 4096 or h > 4096:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Image dimensions ({w}x{h}) exceed maximum allowed 4096x4096 resolution.")
    if w < 10 or h < 10:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Image dimensions ({w}x{h}) are too small for detection.")
        
    return img

def run_rt_detr_inference(img: Image.Image, conf_threshold: float = 0.50, iou_threshold: float = 0.45):
    if conf_threshold < 0.50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confidence threshold must be >= 0.50 to enforce false-positive suppression baseline."
        )
        
    model_inst = get_model()
    start_t = time.time()
    results = model_inst.predict(source=img, conf=conf_threshold, verbose=False)
    infer_ms = (time.time() - start_t) * 1000.0
    
    res = results[0]
    w, h = img.size
    
    raw_detections = []
    for box in res.boxes:
        cls_id = int(box.cls[0])
        cls_name = model_inst.names.get(cls_id, f"class_{cls_id}")
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        
        norm_bbox = [
            round(xyxy[0] / w, 4),
            round(xyxy[1] / h, 4),
            round(xyxy[2] / w, 4),
            round(xyxy[3] / h, 4)
        ]
        
        raw_detections.append({
            "cls_id": cls_id,
            "class": cls_name,
            "confidence": round(conf, 4),
            "bbox": norm_bbox,
            "box": norm_bbox
        })
        
    # Apply strict Python Class-Aware NMS (IoU <= 0.45)
    filtered_detections = apply_class_aware_nms(raw_detections, iou_threshold=iou_threshold)
    
    summary = {}
    for d in filtered_detections:
        summary[d["class"]] = summary.get(d["class"], 0) + 1
        
    return summary, filtered_detections, infer_ms

@app.post("/detect", response_model=DetectResponse, status_code=status.HTTP_200_OK)
async def detect(file: UploadFile = File(...), conf_threshold: float = 0.50):
    """
    Part A: Accepts image upload and returns RT-DETR object detections JSON with strict NMS enforcement (conf >= 0.50).
    """
    logger.info(f"Received /detect request: filename={file.filename}, conf={conf_threshold}")
    file_bytes = await file.read()
    img = validate_and_load_image(file_bytes, filename=file.filename or "")
    
    summary, detections, infer_ms = run_rt_detr_inference(img, conf_threshold=conf_threshold, iou_threshold=0.45)
    
    return {
        "success": True,
        "num_detections": len(detections),
        "summary": summary,
        "detections": detections,
        "inference_time_ms": round(infer_ms, 2)
    }

@app.post("/detect/visualize", response_class=Response, status_code=status.HTTP_200_OK)
async def detect_visualize(file: UploadFile = File(...), conf_threshold: float = 0.50):
    """
    Bonus Visual Endpoint: Returns the image with rendered 2D bounding boxes and class labels.
    """
    file_bytes = await file.read()
    img = validate_and_load_image(file_bytes, filename=file.filename or "")
    w, h = img.size
    
    summary, detections, _ = run_rt_detr_inference(img, conf_threshold=conf_threshold, iou_threshold=0.45)
    
    draw = ImageDraw.Draw(img)
    color_map = {
        "person": "#3B82F6",      # Blue
        "hard-hat": "#EAB308",    # Yellow
        "safety-vest": "#F97316"   # Orange
    }
    
    for d in detections:
        cls_name = d["class"]
        conf = d["confidence"]
        norm_box = d["bbox"]
        
        box = [norm_box[0] * w, norm_box[1] * h, norm_box[2] * w, norm_box[3] * h]
        color = color_map.get(cls_name, "#10B981")
        
        draw.rectangle(box, outline=color, width=4)
        label_text = f"{cls_name} {conf:.2f}"
        draw.rectangle([box[0], max(0, box[1] - 22), box[0] + len(label_text) * 11, box[1]], fill=color)
        draw.text((box[0] + 4, max(0, box[1] - 20)), label_text, fill="white")
        
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/reason", response_model=ReasonResponse, status_code=status.HTTP_200_OK)
async def reason(question: str = Form(...), file: UploadFile = File(...)):
    """
    Part B: Accepts image + question and executes handwritten intent routing, spatial reasoning, and confidence guardrails.
    """
    logger.info(f"Received /reason request: question='{question}', filename={file.filename}")
    
    file_bytes = await file.read()
    img = validate_and_load_image(file_bytes, filename=file.filename or "")
    
    route = handwritten_intent_router(question)
    
    if not route["requires_detector"]:
        logger.info("Intent Router: Unrelated query. Skipping RT-DETR detector.")
        return {
            "success": True,
            "decision": "unrelated_query",
            "insufficient_information": False,
            "detector_called": False,
            "answer": "This question is unrelated to the visual contents of the image.",
            "evidence": {}
        }
        
    summary, detections, infer_ms = run_rt_detr_inference(img, conf_threshold=0.50, iou_threshold=0.45)
    logger.info(f"Detector executed in {infer_ms:.2f}ms with {len(detections)} detections after NMS.")
    
    pipeline_res = process_reasoning_pipeline(question, detections)
    
    return {
        "success": True,
        "decision": pipeline_res["decision"],
        "insufficient_information": pipeline_res["insufficient_information"],
        "detector_called": True,
        "answer": pipeline_res["answer"],
        "evidence": pipeline_res["evidence"]
    }
