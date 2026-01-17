Crackathon: Road Damage Detection 

#Project Overview

This repository contains the complete solution for the Crackathon competition, focused on automated detection and classification of road surface damage. The model is built to identify five types of road damage from the RDD2022 dataset using the YOLOv8 object detection framework.

Final Validation Score: 0.546 mAP50 | 0.287 mAP50-95

#Project Structure

Crackthon/
├── data/
│   ├── train/images/          # Training images
│   ├── train/labels/          # Training labels in YOLO format
│   ├── val/images/            # Validation images  
│   ├── val/labels/            # Validation labels
│   └── test/images/           # Test images (for final prediction)
├── runs/detect/optimized_training/
│   ├── weights/best.pt        # Best trained model weights
│   └── results.csv            # Training metrics and plots
├── predictions/final/labels/  # Generated prediction files
├── submission.zip             # Final competition submission
├── data.yaml                  # Dataset configuration
├── analyze_data.py            # Data analysis and class distribution
└── requirements.txt           # Python dependencies

#Environment Setup

# Create and activate virtual environment
python -m venv crackathon_env
crackathon_env\Scripts\activate  # Windows
# source crackathon_env/bin/activate  # 

# Install dependencies
pip install ultralytics==8.3.246 torch==2.5.1+cu121 torchvision==0.20.1+cu121

#Data Preparation

Download the RDD2022 dataset from the provided links and organize it as shown in the project structure above. Ensure your data.yaml contains:

path: /path/to/data
train: train/images
val: val/images
test: test/images
nc: 5
names: ['Longitudinal crack', 'Transverse crack', 'Alligator crack', 'Other corruption', 'Pothole']

#Data Analysis

Before training, we analyzed the class distribution to understand the dataset characteristics:

Total Training Images: 26,869

Class Distribution:

Longitudinal Crack: 39.3% (18,201 instances)

Transverse Crack: 18.1% (8,386 instances)

Alligator Crack: 16.3% (7,527 instances)

Other Corruption: 16.3% (7,554 instances)

Pothole: 10.0% (4,628 instances)

Key Insight: Significant class imbalance identified, with Pothole being the least represented class (4:1 ratio compared to Longitudinal Crack). This informed our training strategy and explained performance variations across classes.

#Model Training

Hardware Constraints & Strategy
Trained on an NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM). Due to memory limitations:

Selected YOLOv8m (medium) instead of larger variants (l/x)

Reduced image size to 416x416 pixels (from standard 640)

Enabled FP16 mixed precision (half=True) to reduce memory usage

#Training Command

yolo task=detect mode=train \
  model=yolov8m.pt \
  data=data.yaml \
  epochs=50 \
  imgsz=416 \
  batch=4 \
  name=optimized_training \
  patience=20 \
  cos_lr=True \
  device=0 \
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  translate=0.1 \
  scale=0.5 \
  fliplr=0.5 \
  mosaic=1.0 \
  mixup=0.1 \
  half=True

  #Training Details

  Epochs: 50 (with early stopping patience=20)

Batch Size: 4 (limited by GPU memory)

Image Size: 416x416

Optimizer: SGD with cosine learning rate scheduler

Augmentations: Mosaic, MixUp, HSV adjustments, flips, translation, scaling

Training Time: ~13 hours on RTX 3050 Ti

#Results & Performance

Final Validation Metrics
Metric	Score
mAP50	0.546
mAP50-95	0.287
Precision	0.681
Recall	0.531
Per-Class Performance
Damage Class	    AP (mAP50)	Images	Instances
Longitudinal Crack	0.569	2,011	3,800
Transverse Crack	0.600	1,159	1,760
Alligator Crack	    0.565	1,223	1,953
Other Corruption	0.697	1,691	1,853
Pothole	            0.433	544	    965

Key Findings:

Strongest Class: Other Corruption (0.697 AP) - model handled diverse damage types well

Most Challenging: Pothole (0.433 AP) - directly correlates with class imbalance (10% of dataset)

Overall Performance: Balanced detection across crack types with APs between 0.565-0.600

#Generating Predictions & Submission

1.  Create Test Predictions

yolo predict \
  model=runs/detect/optimized_training/weights/best.pt \
  source=data/test/images \
  save_txt=True \
  conf=0.25 \
  project=predictions \
  name=final

2. Create Submission File

# On Windows PowerShell
Compress-Archive -Path "predictions/final/labels/*.txt" -DestinationPath "submission.zip" -Force
Submission Structure
Ensure submission.zip contains only .txt files (no subfolders):

submission.zip
├── test_image_001.txt
├── test_image_002.txt
├── test_image_003.txt
└── ...
Each .txt file follows YOLO format:

<class_id> <x_center> <y_center> <width> <height> <confidence>

#Model Performance Analysis

Strengths
Robust to "Other Corruption": Highest AP (0.697) despite class diversity

Hardware-Efficient: Achieved competitive results on 4GB GPU through strategic compromises

Reproducible Pipeline: Complete from data analysis to submission generation

Limitations & Challenges
GPU Memory Constraints: Limited model size (YOLOv8m) and image resolution (416px)

Class Imbalance: Pothole class significantly underrepresented affecting its AP

Trade-offs: Lower resolution may affect detection of very fine cracks

Potential Improvements
Given more resources:

Larger Model: YOLOv8l with 640px images on GPU with ≥8GB VRAM

Class Balancing: Oversampling or loss weighting for Pothole class

Advanced Augmentations: Copy-paste augmentation for rare classes

Hyperparameter Tuning: Systematic search for optimal loss weights

#Technical Specifications


Component	Specification
Framework	Ultralytics YOLOv8
Base Model	YOLOv8m (pretrained on COCO)
Input Resolution	416x416 pixels
GPU	NVIDIA GeForce RTX 3050 Ti (4GB)
Training Time	~13 hours for 50 epochs
Final Model Size	52.8 MB (best.pt)


#Reproducibility Notes

Environment: Use exact package versions in requirements.txt

Data: Use the randomized RDD2022 dataset from official links

Hardware: Results may vary with different GPU memory constraints

Random Seeds: Fixed seeds not explicitly set; slight variations possible
