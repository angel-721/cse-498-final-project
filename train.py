import torch
import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np


def setup_dataset_path():
    """
    Set up the dataset path for training.
    NOTE: Will only set the dataset path for Windows OS.
    Returns:
        str: The path to the dataset YAML file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = current_dir + "\\datasets\\cards\\data.yaml"
    return dataset


if __name__ == '__main__':
    # Check if GPU is available and set device accordingly
    print("Checking for GPU...")
    is_cuda_available = torch.cuda.is_available()
    print(is_cuda_available)
    if is_cuda_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    dataset = setup_dataset_path()

    # Load a COCO-pretrained YOLOv11n model
    model = YOLO("./models/yolo11n.pt")

    results = model.train(data=dataset, epochs=20, imgsz=640, device=device)
