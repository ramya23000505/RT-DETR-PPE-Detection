import re
import json
import logging
from app.llm_client import call_gemini_reasoning

logger = logging.getLogger("rt_detr_ppe.reasoning")

PPE_KEYWORDS = [
    "hardhat", "hard-hat", "helmet", "head protection",
    "vest", "safety-vest", "jacket", "reflective",
    "person", "people", "worker", "man", "woman", "human", "someone", "anyone", "everyone",
    "object", "detect", "see", "show", "count", "present", "wear", "wearing", "missing", "safety", "audit", "compliance"
]

OUT_OF_VOCABULARY_KEYWORDS = [
    "shoe", "shoes", "boot", "boots", "glove", "gloves", "glass", "glasses", "goggle", "goggles",
    "mask", "pant", "pants", "shirt", "car", "truck", "dog", "cat", "color", "colors"
]

def handwritten_intent_router(question: str) -> dict:
    q_lower = question.lower().strip()
    contains_keyword = any(kw in q_lower for kw in PPE_KEYWORDS)
    out_of_scope_indicators = [
        "capital of", "weather today", "who wrote", "president of", "currency of",
        "prime minister", "square root", "formula for", "definition of"
    ]
    is_out_of_scope = any(indicator in q_lower for indicator in out_of_scope_indicators)
    
    if is_out_of_scope or not contains_keyword:
        return {
            "decision": "unrelated_query",
            "requires_detector": False,
            "reason": "Question is out-of-scope or unrelated to image visual content."
        }
        
    return {
        "decision": "detection_required",
        "requires_detector": True,
        "reason": "Question asks about objects, counts, or PPE safety compliance."
    }

def analyze_geometric_relationships(detections: list) -> dict:
    persons = []
    hard_hats = []
    safety_vests = []
    
    for idx, obj in enumerate(detections):
        cls_name = obj["class"].lower()
        box = obj["box"]
        
        if cls_name == "person":
            persons.append({"id": idx, "box": box})
        elif cls_name in ["hard-hat", "hardhat", "helmet"]:
            hard_hats.append({"id": idx, "box": box})
        elif cls_name in ["safety-vest", "vest"]:
            safety_vests.append({"id": idx, "box": box})
            
    person_relationships = []
    
    for p in persons:
        px1, py1, px2, py2 = p["box"]
        pw = px2 - px1
        ph = py2 - py1
        
        hh_assoc_region = [px1 - pw*0.1, py1 - ph*0.2, px2 + pw*0.1, py1 + ph*0.35]
        vest_assoc_region = [px1 - pw*0.1, py1 + ph*0.15, px2 + pw*0.1, py1 + ph*0.75]
        
        has_hh = False
        has_vest = False
        
        for hh in hard_hats:
            hx1, hy1, hx2, hy2 = hh["box"]
            hcx = (hx1 + hx2) / 2.0
            hcy = (hy1 + hy2) / 2.0
            if (hh_assoc_region[0] <= hcx <= hh_assoc_region[2]) and (hh_assoc_region[1] <= hcy <= hh_assoc_region[3]):
                has_hh = True
                break
                
        for v in safety_vests:
            vx1, vy1, vx2, vy2 = v["box"]
            vcx = (vx1 + vx2) / 2.0
            vcy = (vy1 + vy2) / 2.0
            if (vest_assoc_region[0] <= vcx <= vest_assoc_region[2]) and (vest_assoc_region[1] <= vcy <= vest_assoc_region[3]):
                has_vest = True
                break
                
        person_relationships.append({
            "person_id": p["id"],
            "person_box": p["box"],
            "has_hard_hat": has_hh,
            "has_safety_vest": has_vest
        })
        
    return {
        "total_persons": len(persons),
        "total_hard_hats": len(hard_hats),
        "total_safety_vests": len(safety_vests),
        "person_compliance": person_relationships
    }

def confidence_and_evidence_guardrail(question: str, detections: list, rel_analysis: dict, min_confidence: float = 0.35) -> dict:
    valid_detections = [d for d in detections if d.get("confidence", 0.0) >= min_confidence]
    if len(valid_detections) == 0:
        return {
            "sufficient": False,
            "reason": "Insufficient information to answer confidently. No objects detected with sufficient confidence in the image."
        }
        
    q_lower = question.lower()
    
    for oov in OUT_OF_VOCABULARY_KEYWORDS:
        if oov in q_lower:
            return {
                "sufficient": False,
                "reason": f"Insufficient information to answer confidently. Asking about '{oov}' which is outside the model detection bounding box class taxonomy (person, hard-hat, safety-vest)."
            }
            
    if any(k in q_lower for k in ["helmet", "hard-hat", "hard hat"]):
        if rel_analysis["total_persons"] == 0 and rel_analysis["total_hard_hats"] == 0:
            return {
                "sufficient": False,
                "reason": "Insufficient information to answer confidently. Neither persons nor hard hats were detected."
            }
            
    avg_conf = sum(d["confidence"] for d in valid_detections) / len(valid_detections)
    if avg_conf < 0.40 and len(valid_detections) < 2:
        return {
            "sufficient": False,
            "reason": "Insufficient information to answer confidently. Detection confidence is too low to guarantee accuracy."
        }
        
    return {
        "sufficient": True,
        "reason": "Evidence is sufficient for reasoning."
    }

def process_reasoning_pipeline(question: str, detections: list) -> dict:
    route = handwritten_intent_router(question)
    if not route["requires_detector"]:
        return {
            "decision": "unrelated_query",
            "insufficient_information": False,
            "detector_called": False,
            "answer": "This question is unrelated to the visual contents of the image.",
            "evidence": {}
        }
        
    rel_analysis = analyze_geometric_relationships(detections)
    guardrail = confidence_and_evidence_guardrail(question, detections, rel_analysis)
    if not guardrail["sufficient"]:
        return {
            "decision": "insufficient_information",
            "insufficient_information": True,
            "detector_called": True,
            "answer": guardrail["reason"],
            "evidence": {
                "detections": detections,
                "spatial_relationships": rel_analysis
            }
        }
        
    system_prompt = """You are a precise computer vision safety reasoning assistant.
Provide a granular, audit-ready safety breakdown based ONLY on the supplied evidence."""

    person_compliance = rel_analysis["person_compliance"]
    total_workers = len(person_compliance)
    fully_compliant = sum(1 for p in person_compliance if p["has_hard_hat"] and p["has_safety_vest"])
    has_vest_only = sum(1 for p in person_compliance if not p["has_hard_hat"] and p["has_safety_vest"])
    has_hat_only = sum(1 for p in person_compliance if p["has_hard_hat"] and not p["has_safety_vest"])
    non_compliant = sum(1 for p in person_compliance if not p["has_hard_hat"] and not p["has_safety_vest"])

    audit_answer = (
        f"Safety Audit Complete: Detected {total_workers} worker(s). "
        f"Fully Compliant: {fully_compliant}. "
        f"Partial Compliance: {has_vest_only} (wearing vest only, missing hard-hat), "
        f"{has_hat_only} (wearing hard-hat only, missing vest). "
        f"Non-Compliant: {non_compliant}."
    )

    evidence_str = json.dumps({
        "audit_summary": audit_answer,
        "detections_summary": {
            "persons": rel_analysis["total_persons"],
            "hard_hats": rel_analysis["total_hard_hats"],
            "safety_vests": rel_analysis["total_safety_vests"]
        },
        "person_ppe_associations": person_compliance
    }, indent=2)

    prompt = f"User Question: {question}\n\nStructured Evidence:\n{evidence_str}"
    
    llm_answer = call_gemini_reasoning(prompt=prompt, system_instruction=system_prompt)
    
    if "Gemini API Key missing" in llm_answer or "LLM Service" in llm_answer:
        llm_answer = audit_answer

    return {
        "decision": "structured_reasoning_success",
        "insufficient_information": False,
        "detector_called": True,
        "answer": llm_answer,
        "evidence": {
            "detections": detections,
            "spatial_relationships": rel_analysis
        }
    }
