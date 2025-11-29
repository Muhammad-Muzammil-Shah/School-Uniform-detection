# School-Uniform-detection

## Overview
This project demonstrates object detection for school uniforms using YOLOv5 and YOLO11 models. It includes Jupyter notebooks for training, evaluating, and running inference on custom datasets, with step-by-step instructions and code examples.

## How It Works
- **Custom Dataset Preparation:** Uses Roboflow to collect, annotate, and export images for training.
- **Model Training:** Notebooks show how to train YOLOv5 and YOLO11 models on the dataset, including hyperparameter tuning and augmentation.
- **Evaluation:** Visualizes training results, confusion matrix, and sample predictions.
- **Inference:** Runs object detection on images and videos, saving results for review.
- **Deployment:** Instructions for deploying models using Roboflow and running inference on edge devices.

## Main Files
- `train_yolov5_object_detection_on_custom_data.ipynb`: Step-by-step guide for YOLOv5 training and evaluation.
- `YOLO11_Uniform_Detection.ipynb`: Step-by-step guide for YOLO11 training and inference.
- `MergedImages.png`: Example image for inference.
- `notebooks/`: Additional notebooks and resources.

## Usage
1. Prepare your dataset using Roboflow and export in YOLO format.
2. Follow the notebooks to train and evaluate your model.
3. Run inference on new images or videos to detect uniforms.
4. Deploy your model using Roboflow or on local/edge devices.

## Technologies
- Python, Jupyter Notebook
- YOLOv5, YOLO11 (Ultralytics)
- Roboflow (dataset management)

This repo is ideal for learning and experimenting with custom object detection workflows.