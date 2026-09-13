import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app, compute_iou, apply_class_aware_nms

client = TestClient(app)

def create_dummy_image_bytes(size=(640, 640), color="blue"):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_compute_iou_identical_boxes():
    box1 = [0.1, 0.1, 0.5, 0.5]
    box2 = [0.1, 0.1, 0.5, 0.5]
    assert compute_iou(box1, box2) == 1.0

def test_compute_iou_disjoint_boxes():
    box1 = [0.0, 0.0, 0.2, 0.2]
    box2 = [0.5, 0.5, 0.8, 0.8]
    assert compute_iou(box1, box2) == 0.0

def test_class_aware_nms_filters_duplicates():
    detections = [
        {"cls_id": 1, "class": "hard-hat", "confidence": 0.85, "bbox": [0.1, 0.1, 0.4, 0.4]},
        {"cls_id": 1, "class": "hard-hat", "confidence": 0.70, "bbox": [0.11, 0.11, 0.41, 0.41]}, # Duplicate IoU > 0.8
        {"cls_id": 0, "class": "person", "confidence": 0.90, "bbox": [0.1, 0.1, 0.5, 0.9]}
    ]
    filtered = apply_class_aware_nms(detections, iou_threshold=0.45)
    assert len(filtered) == 2
    # Ensure highest confidence hard-hat (0.85) was kept and 0.70 was removed
    hh_dets = [d for d in filtered if d["class"] == "hard-hat"]
    assert len(hh_dets) == 1
    assert hh_dets[0]["confidence"] == 0.85

def test_detect_endpoint_valid_image():
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/detect?conf_threshold=0.50",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "num_detections" in data
    assert "summary" in data
    assert "detections" in data

def test_detect_endpoint_rejects_low_conf_threshold():
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/detect?conf_threshold=0.30",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 400
    assert "Confidence threshold must be >= 0.50" in response.json()["detail"]

def test_detect_visualize_endpoint():
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/detect/visualize?conf_threshold=0.50",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

def test_detect_endpoint_invalid_file():
    response = client.post(
        "/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400

def test_reason_endpoint_unrelated_query():
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/reason",
        data={"question": "What is the capital of France?"},
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["decision"] == "unrelated_query"
    assert data["detector_called"] is False
    assert "unrelated" in data["answer"].lower()

def test_reason_endpoint_detection_required():
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/reason",
        data={"question": "How many persons and hard-hats are detected?"},
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["detector_called"] is True
