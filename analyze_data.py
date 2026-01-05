# analyze_data.py
import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

def analyze_class_distribution():
    """Analyzes and plots the distribution of damage classes in the training set."""
    
    # Paths (ensure these match your structure)
    label_dir = "D:\\Crackthon\\data\\train\\labels"
    class_names = {
        0: "Longitudinal Crack",
        1: "Transverse Crack",
        2: "Alligator Crack", 
        3: "Other Corruption",  # The challenging class
        4: "Pothole"            # Often the rarest
    }
    
    # 1. Count instances per class
    class_counter = Counter()
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    print(f"📈 Analyzing {len(label_files)} label files...")
    
    for l_file in label_files:
        with open(os.path.join(label_dir, l_file), 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    class_id = int(line.split()[0])
                    class_counter[class_id] += 1
    
    # 2. Print the raw statistics
    print("\n" + "="*50)
    print("DAMAGE CLASS DISTRIBUTION (Training Set)")
    print("="*50)
    
    total_instances = sum(class_counter.values())
    for class_id in sorted(class_counter.keys()):
        count = class_counter[class_id]
        percentage = (count / total_instances) * 100
        print(f"{class_names[class_id]} (Class {class_id}): {count:>5} instances ({percentage:.1f}%)")
    
    print(f"\n{'Total:':<25} {total_instances:>5} instances")
    
    # 3. Create a bar chart for visualization
    plt.figure(figsize=(10, 6))
    classes = [class_names[i] for i in sorted(class_counter.keys())]
    counts = [class_counter[i] for i in sorted(class_counter.keys())]
    
    bars = plt.bar(classes, counts, color=['skyblue', 'lightgreen', 'gold', 'orange', 'lightcoral'])
    plt.title('Road Damage Class Distribution in Training Set', fontsize=14, fontweight='bold')
    plt.xlabel('Damage Class', fontsize=12)
    plt.ylabel('Number of Instances', fontsize=12)
    plt.xticks(rotation=15)
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*total_instances,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 4. Calculate and display key insights
    print("\n" + "="*50)
    print("KEY INSIGHTS FOR TRAINING STRATEGY")
    print("="*50)
    
    most_common_class = max(class_counter, key=class_counter.get)
    least_common_class = min(class_counter, key=class_counter.get)
    
    imbalance_ratio = class_counter[most_common_class] / class_counter[least_common_class]
    
    print(f"1. Most common class: '{class_names[most_common_class]}'")
    print(f"2. Least common class: '{class_names[least_common_class]}'")
    print(f"3. Imbalance ratio: {imbalance_ratio:.1f}x (Most/Least)")
    print(f"4. 'Other Corruption' has {class_counter[3]} instances.")
    
    # 5. Training recommendations based on the data
    print("\n" + "="*50)
    print("RECOMMENDED TRAINING ACTIONS")
    print("="*50)
    
    if imbalance_ratio > 10:
        print("  HIGH IMBALANCE DETECTED!")
        print("   → You MUST use class-weighted loss in training.")
        print("   → Focus augmentation on '{class_names[least_common_class]}'.")
    
    if class_counter[4] < 1000:  # Pothole threshold
        print("  POTHOLE CLASS IS VERY RARE!")
        print("   → Consider mosaic/mixup augmentation specifically for potholes.")
    
    print("   → For 'Other Corruption': Use diverse augmentation (blur, noise, contrast).")
    print("   → Monitor per-class accuracy, not just overall mAP.")
    
    return class_counter

# Run the analysis
if __name__ == "__main__":

    analyze_class_distribution()
