import os
import argparse
import yaml
import json
from ultralytics import RTDETR

def evaluate_model(weights_path="runs/rtdetr_ppe_final/weights/best.pt", data_yaml="dataset/dataset.yaml", split="test"):
    print("=" * 60)
    print(f"EVALUATING RT-DETR MODEL ON HELD-OUT '{split.upper()}' SET")
    print("=" * 60)
    
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Fine-tuned PPE weights not found at {weights_path}. "
            "Evaluation must not silently fall back to a COCO checkpoint."
        )
        
    model = RTDETR(weights_path)
    
    # Run evaluation on test set split
    results = model.val(data=data_yaml, split=split, save_json=True, project="runs", name="evaluate_test", exist_ok=True)
    
    with open(data_yaml, "r") as f:
        meta = yaml.safe_load(f)
    class_names = meta.get("names", {0: "person", 1: "hard-hat", 2: "safety-vest"})
    
    map50_95 = float(results.box.map)
    map50 = float(results.box.map50)
    mp = float(results.box.mp)
    mr = float(results.box.mr)
    
    print("\n--- TEST SET METRICS ---")
    print(f"mAP50-95          : {map50_95:.4f}")
    print(f"mAP50             : {map50:.4f}")
    print(f"Mean Precision    : {mp:.4f}")
    print(f"Mean Recall       : {mr:.4f}")
    
    per_class_metrics = {}
    if hasattr(results.box, 'maps') and results.box.maps is not None:
        for idx, name in class_names.items():
            if idx < len(results.box.maps):
                per_class_metrics[name] = float(results.box.maps[idx])
                print(f"  - Class '{name}' mAP50-95: {results.box.maps[idx]:.4f}")
                
    eval_summary = {
        "weights": weights_path,
        "split": split,
        "mAP50_95": map50_95,
        "mAP50": map50,
        "precision": mp,
        "recall": mr,
        "per_class": per_class_metrics
    }
    
    out_json = os.path.join("runs", "evaluate_test", "metrics.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(eval_summary, f, indent=2)
        
    print(f"\nSaved evaluation metrics to: {out_json}")
    return eval_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR model")
    parser.add_argument("--weights", type=str, default="runs/rtdetr_ppe_final/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--data", type=str, default="dataset/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--split", type=str, default="test", help="Split name ('test' or 'val')")
    args = parser.parse_args()
    
    evaluate_model(args.weights, args.data, args.split)
