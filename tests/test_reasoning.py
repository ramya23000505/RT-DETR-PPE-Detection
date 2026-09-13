import pytest
from app.reasoning import (
    handwritten_intent_router,
    analyze_geometric_relationships,
    confidence_and_evidence_guardrail,
    process_reasoning_pipeline
)

def test_intent_router_unrelated_query():
    q = "What is the capital of France?"
    res = handwritten_intent_router(q)
    assert res["requires_detector"] is False
    assert res["decision"] == "unrelated_query"

def test_intent_router_detection_query():
    q = "How many people are wearing a hard-hat?"
    res = handwritten_intent_router(q)
    assert res["requires_detector"] is True
    assert res["decision"] == "detection_required"

def test_geometric_relationship_association():
    detections = [
        {"class": "person", "confidence": 0.90, "box": [0.1, 0.2, 0.4, 0.8]},
        {"class": "hard-hat", "confidence": 0.85, "box": [0.12, 0.18, 0.38, 0.30]},
        {"class": "safety-vest", "confidence": 0.88, "box": [0.11, 0.35, 0.39, 0.60]}
    ]
    rel = analyze_geometric_relationships(detections)
    assert rel["total_persons"] == 1
    assert rel["total_hard_hats"] == 1
    assert rel["total_safety_vests"] == 1
    p_compliance = rel["person_compliance"][0]
    assert p_compliance["has_hard_hat"] is True
    assert p_compliance["has_safety_vest"] is True

def test_confidence_guardrail_zero_detections():
    detections = []
    rel = analyze_geometric_relationships(detections)
    guard = confidence_and_evidence_guardrail("Is anyone wearing a helmet?", detections, rel)
    assert guard["sufficient"] is False
    assert "Insufficient information" in guard["reason"]

def test_confidence_guardrail_out_of_vocabulary():
    detections = [
        {"class": "person", "confidence": 0.95, "box": [0.1, 0.1, 0.5, 0.5]}
    ]
    rel = analyze_geometric_relationships(detections)
    guard = confidence_and_evidence_guardrail("What color shoes is the worker wearing?", detections, rel)
    assert guard["sufficient"] is False
    assert "outside the model detection bounding box class taxonomy" in guard["reason"]

def test_confidence_guardrail_color_attribute():
    detections = [
        {"class": "safety-vest", "confidence": 0.92, "box": [0.1, 0.1, 0.5, 0.5]}
    ]
    rel = analyze_geometric_relationships(detections)
    guard = confidence_and_evidence_guardrail("What color safety vest is the worker wearing?", detections, rel)
    assert guard["sufficient"] is False
    assert "Asking about 'color'" in guard["reason"]

def test_full_pipeline_unrelated_query():
    res = process_reasoning_pipeline("What is the weather today?", [])
    assert res["decision"] == "unrelated_query"
    assert res["detector_called"] is False
    assert "unrelated" in res["answer"].lower()

def test_full_pipeline_oov_query():
    detections = [
        {"class": "person", "confidence": 0.95, "box": [0.1, 0.1, 0.5, 0.5]}
    ]
    res = process_reasoning_pipeline("What color shoes are they wearing?", detections)
    assert res["decision"] == "insufficient_information"
    assert res["insufficient_information"] is True
    assert res["detector_called"] is True
    assert "Insufficient information" in res["answer"]

def test_full_pipeline_compliant_query():
    detections = [
        {"class": "person", "confidence": 0.95, "box": [0.1, 0.2, 0.5, 0.9]},
        {"class": "hard-hat", "confidence": 0.88, "box": [0.12, 0.18, 0.48, 0.32]},
        {"class": "safety-vest", "confidence": 0.90, "box": [0.11, 0.35, 0.49, 0.65]}
    ]
    res = process_reasoning_pipeline("Is everyone wearing a hard hat?", detections)
    assert res["decision"] == "structured_reasoning_success"
    assert res["insufficient_information"] is False
    assert res["detector_called"] is True
    assert "Safety Audit Complete" in res["answer"] or "Fully Compliant" in res["answer"]
