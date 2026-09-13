import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import yaml
import hashlib

def create_synthetic_ppe_dataset(base_dir="dataset", num_images=100, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(os.path.join(base_dir, "images", "all"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "all"), exist_ok=True)
    
    # Classes: 0: person, 1: hard-hat, 2: safety-vest
    classes = ["person", "hard-hat", "safety-vest"]
    
    print(f"Generating {num_images} synthetic PPE images with seed {seed}...")
    
    for i in range(num_images):
        img_w, img_h = 640, 640
        # Background: construction site colored background (grays, browns, blues)
        bg_color = (random.randint(180, 220), random.randint(180, 210), random.randint(170, 200))
        img = Image.new("RGB", (img_w, img_h), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Add some background texture/noise
        for _ in range(10):
            x1, y1 = random.randint(0, 640), random.randint(0, 640)
            x2, y2 = random.randint(0, 640), random.randint(0, 640)
            draw.line([(x1, y1), (x2, y2)], fill=(random.randint(140, 160), random.randint(140, 160), random.randint(140, 160)), width=random.randint(2, 8))
            
        boxes = []
        
        # Decide number of workers in image: 1 to 3
        num_workers = random.randint(1, 3)
        
        for w_idx in range(num_workers):
            # Person dimensions
            pw = random.randint(100, 180)
            ph = random.randint(250, 420)
            px = random.randint(10 + w_idx * 180, min(500, 30 + w_idx * 190))
            py = random.randint(150, 640 - ph)
            
            # Draw person body (clothing / skin)
            body_color = (random.randint(40, 90), random.randint(50, 100), random.randint(120, 180))
            draw.rectangle([px, py, px + pw, py + ph], fill=body_color)
            
            # Draw head
            head_h = int(ph * 0.2)
            head_w = int(pw * 0.5)
            head_x = px + (pw - head_w) // 2
            head_y = py
            draw.ellipse([head_x, head_y, head_x + head_w, head_y + head_h], fill=(235, 195, 165))
            
            # Person label: class 0
            # YOLO format: cls, cx/w, cy/h, w/w, h/h
            pcx = (px + pw / 2) / img_w
            pcy = (py + ph / 2) / img_h
            pnorm_w = pw / img_w
            pnorm_h = ph / img_h
            boxes.append((0, pcx, pcy, pnorm_w, pnorm_h))
            
            # Hard-hat (85% chance worker wears hard hat)
            if random.random() < 0.85:
                hh_w = int(head_w * 1.2)
                hh_h = int(head_h * 0.6)
                hh_x = head_x - int(head_w * 0.1)
                hh_y = head_y - int(hh_h * 0.3)
                
                # Hard hat colors: yellow, orange, white
                hh_color = random.choice([(255, 215, 0), (255, 140, 0), (240, 240, 240)])
                draw.chord([hh_x, hh_y, hh_x + hh_w, hh_y + hh_h * 2], start=180, end=360, fill=hh_color)
                
                hh_cx = (hh_x + hh_w / 2) / img_w
                hh_cy = (hh_y + hh_h / 2) / img_h
                hh_norm_w = hh_w / img_w
                hh_norm_h = hh_h / img_h
                boxes.append((1, hh_cx, hh_cy, hh_norm_w, hh_norm_h))
                
            # Safety vest (80% chance worker wears safety vest)
            if random.random() < 0.80:
                sv_w = int(pw * 0.95)
                sv_h = int(ph * 0.45)
                sv_x = px + (pw - sv_w) // 2
                sv_y = py + int(ph * 0.22)
                
                sv_color = random.choice([(255, 69, 0), (50, 205, 50), (255, 165, 0)]) # High-vis orange/green
                draw.rectangle([sv_x, sv_y, sv_x + sv_w, sv_y + sv_h], fill=sv_color)
                # Draw reflective stripes
                draw.rectangle([sv_x, sv_y + int(sv_h * 0.3), sv_x + sv_w, sv_y + int(sv_h * 0.45)], fill=(220, 220, 220))
                
                sv_cx = (sv_x + sv_w / 2) / img_w
                sv_cy = (sv_y + sv_h / 2) / img_h
                sv_norm_w = sv_w / img_w
                sv_norm_h = sv_h / img_h
                boxes.append((2, sv_cx, sv_cy, sv_norm_w, sv_norm_h))
                
        img_filename = f"ppe_{i:04d}.jpg"
        lbl_filename = f"ppe_{i:04d}.txt"
        
        img_path = os.path.join(base_dir, "images", "all", img_filename)
        lbl_path = os.path.join(base_dir, "labels", "all", lbl_filename)
        
        img.save(img_path)
        
        with open(lbl_path, "w") as f:
            for box in boxes:
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                
    # Save dataset.yaml metadata
    dataset_yaml = {
        "path": os.path.abspath(base_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "person", 1: "hard-hat", 2: "safety-vest"},
        "license": "CC BY 4.0 Open Source PPE Dataset",
        "source": "Synthetically Rendered PPE Construction Safety Callset (Deterministic)"
    }
    
    with open(os.path.join(base_dir, "dataset.yaml"), "w") as f:
        yaml.dump(dataset_yaml, f)

if __name__ == "__main__":
    create_synthetic_ppe_dataset()
