"""
CORRECTED: Generate competition predictions with confidence scores
"""

import os
import zipfile
from ultralytics import YOLO
from pathlib import Path

def generate_correct_predictions():
    """
    Generate predictions with confidence scores (6 values per line)
    """
    # Load your trained model
    model_path = "runs/detect/optimized_training/weights/best.pt"
    model = YOLO(model_path)
    
    # Test images directory
    test_dir = "data/test/images"
    
    # Create predictions directory
    pred_dir = Path("predictions")
    pred_dir.mkdir(exist_ok=True)
    
    print(f"Processing test images from: {test_dir}")
    
    # Process each test image
    for img_file in Path(test_dir).glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Run prediction
            results = model(
                str(img_file),
                conf=0.25,      # Confidence threshold
                iou=0.6,        # IOU threshold
                verbose=False,
                imgsz=416       # Same as training
            )
            
            # Output filename
            txt_file = pred_dir / f"{img_file.stem}.txt"
            
            # Write predictions in CORRECT format (6 values)
            with open(txt_file, 'w') as f:
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        # Extract values
                        class_id = int(box.cls[0].item())        # Class ID
                        confidence = float(box.conf[0].item())   # Confidence
                        x_center, y_center, width, height = box.xywhn[0].tolist()  # Coordinates
                        
                        # Write in required format: class x_center y_center width height confidence
                        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}"
                        f.write(line + "\n")
            
            print(f"  Processed: {img_file.name}")
    
    # Create submission.zip
    print("\nCreating submission.zip...")
    with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for txt_file in pred_dir.glob("*.txt"):
            # Add to zip without folder structure
            zipf.write(txt_file, txt_file.name)
    
    print("\n✅ DONE!")
    print(f"Predictions saved in: {pred_dir}/")
    print(f"Submission file: submission.zip")
    
    # Show example of first prediction
    example_file = next(pred_dir.glob("*.txt"), None)
    if example_file:
        with open(example_file, 'r') as f:
            first_line = f.readline().strip()
            print(f"\nExample prediction line: {first_line}")
            values = first_line.split()
            print(f"Number of values: {len(values)} (should be 6)")
            if len(values) == 6:
                print("✓ Format is CORRECT!")
            else:
                print("✗ Format is INCORRECT!")

def verify_predictions():
    """
    Verify all prediction files have correct format
    """
    pred_dir = Path("predictions")
    
    if not pred_dir.exists():
        print("No predictions folder found!")
        return
    
    errors = []
    
    for txt_file in pred_dir.glob("*.txt"):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                values = line.strip().split()
                if len(values) != 6:
                    errors.append(f"{txt_file.name}: Line {i+1} has {len(values)} values")
                else:
                    # Check confidence is between 0-1
                    try:
                        conf = float(values[5])
                        if not (0 <= conf <= 1):
                            errors.append(f"{txt_file.name}: Line {i+1} confidence {conf} not in [0,1]")
                    except:
                        errors.append(f"{txt_file.name}: Line {i+1} invalid confidence")
    
    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors)-10} more errors")
    else:
        print("\n✅ All prediction files have correct format (6 values per line)")

if __name__ == "__main__":
    print("="*60)
    print("GENERATING COMPETITION PREDICTIONS")
    print("Format: class x_center y_center width height confidence")
    print("="*60)
    
    generate_correct_predictions()
    verify_predictions()