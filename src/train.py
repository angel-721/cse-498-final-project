import torch
import os
from ultralytics import YOLO
import time

EPOCHS = 60

def setup_dataset_path():
    """
    Set up the dataset path for training.
    NOTE: Will only set the dataset path for Windows OS.
    Returns:
        str: The path to the dataset YAML file.
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # go up one directory since the script is in the src, not in the root
    current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

    return current_dir + "\\datasets\\cards\\data.yaml"

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

    working_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Working directory: {working_dir}")
    print(f"Dataset path: {dataset}")

    # Load a COCO-pretrained YOLOv11n model
    model = YOLO("./models/yolo11n.pt")

    # Train the model
    print("Starting training...")
    start_time = time.time()
    results = model.train(data=dataset, epochs=EPOCHS, imgsz=640, device=device)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # Save the model
    model.save(f"./models/yolo11n-cards-{EPOCHS}-epochs.pt")
