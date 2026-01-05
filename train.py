# train.py - ACTUAL TRAINING SCRIPT
"""
Training script for Crackathon competition.
Reproduces the exact training process used.
"""

import argparse
from ultralytics import YOLO
import yaml

def train_model(config_path='data.yaml', epochs=50, imgsz=416, batch=4):
    """
    Train YOLOv8 model on RDD2022 dataset
    """
    print("🚀 Starting training for Crackathon submission...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Dataset: {config['path']}")
    print(f"Classes: {config['names']}")
    
    # Load pretrained model
    model = YOLO('yolov8m.pt')
    
    # Train with competition parameters
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,
        cos_lr=True,
        device=0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        half=True,
        name='crackathon_training',
        verbose=True
    )
    
    print("✅ Training completed!")
    print(f"Model saved to: runs/detect/crackathon_training/weights/best.pt")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=416, help='Image size')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    
    args = parser.parse_args()
    train_model(args.config, args.epochs, args.imgsz, args.batch)