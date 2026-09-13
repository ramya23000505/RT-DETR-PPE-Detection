import os
import time
import argparse
import yaml
import torch
from ultralytics import RTDETR

def train_model(data_yaml_path="dataset/dataset.yaml", epochs=15, batch_size=4, imgsz=640, device='cpu', seed=42):
    """
    Trains/Fine-tunes RT-DETR on custom PPE dataset.
    Logs hyperparameters, hardware, and duration.
    """
    print("=" * 60)
    print("RT-DETR TRAINING & FINE-TUNING PIPELINE")
    print("=" * 60)
    
    start_time = time.time()
    
    # Verify dataset yaml
    with open(data_yaml_path, 'r') as f:
        meta = yaml.safe_load(f)
        
    print(f"Dataset Path  : {meta.get('path')}")
    print(f"Classes       : {meta.get('names')}")
    print(f"Target Epochs : {epochs}")
    print(f"Batch Size    : {batch_size}")
    print(f"Image Size    : {imgsz}")
    print(f"Device        : {device}")
    print(f"Random Seed   : {seed}")
    print(f"PyTorch Ver   : {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Initialize RT-DETR base checkpoint (COCO/ImageNet pretrained backbone for fine-tuning)
    # Pretrained checkpoint ambiguity: Hand-tuned on task-specific dataset, starting from general base model
    model = RTDETR('rtdetr-l.pt')
    
    # Train
# Train with enhanced regularization to prevent false-positive floating boxes
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project='runs',
        name='rtdetr_ppe_final',
        exist_ok=True,
        seed=seed,
        box=7.5,       # Increase box loss gain to penalize loose/misaligned bounding boxes
        cls=1.0,       # Class loss weight for classification accuracy
        degrees=10.0,  # Slight rotation augmentation to handle varied orientations
        mosaic=1.0,    # Use mosaic augmentation for complex background context
        verbose=True
    )
    
    duration = time.time() - start_time
    print(f"\nTraining completed in {duration:.2f} seconds ({duration/60:.2f} minutes).")
    weights_path = os.path.join("runs", "rtdetr_ppe_final", "weights", "best.pt")
    print(f"Best model weights saved to: {weights_path}")
    
    return results, weights_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RT-DETR on PPE dataset")
    parser.add_argument("--data", type=str, default="dataset/dataset.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or 0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    train_model(
        data_yaml_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed
    )
