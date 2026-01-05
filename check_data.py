# check_data.py
import cv2
import os
import matplotlib.pyplot as plt

def check_one_sample():
    # 1. Define paths (adjust if your structure differs)
    train_img_dir = "D:\\Crackthon\\data\\train\\images"
    train_label_dir = "D:\\Crackthon\\data\\train\\labels"
    
    # 2. Get first image file
    img_files = os.listdir(train_img_dir)
    if not img_files:
        print(" No images found! Check your data path.")
        return
    
    first_image = img_files[0]
    print(f"📸 Checking: {first_image}")
    
    # 3. Load and display image
    img_path = os.path.join(train_img_dir, first_image)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to normal colors
    
    # 4. Load corresponding label file
    label_file = first_image.replace(".jpg", ".txt").replace(".png", ".txt")
    label_path = os.path.join(train_label_dir, label_file)
    
    damage_types = {
        0: "Longitudinal crack",
        1: "Transverse crack", 
        2: "Alligator crack",
        3: "Other corruption",
        4: "Pothole"
    }
    
    print("\n🔍 Damage annotations found:")
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                # YOLO format: class x_center y_center width height
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert YOLO coordinates to pixel values
                    h, w = img.shape[:2]
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, damage_types[class_id], (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    print(f"   - {damage_types[class_id]} at [{x1}, {y1}, {x2}, {y2}]")
    else:
        print("   No label file found for this image")
    
    # 5. Display the image with boxes
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f"Sample: {first_image} with annotations")
    plt.axis('off')
    plt.show()
    
    # 6. Quick dataset stats
    print(f"\n Quick stats:")
    print(f"   Total training images: {len(img_files)}")
    print(f"   Total label files: {len(os.listdir(train_label_dir))}")
    
    return True

if __name__ == "__main__":

    check_one_sample()
