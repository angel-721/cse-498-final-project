from ultralytics import YOLO
from train import setup_dataset_path

def main():
    dataset = setup_dataset_path()

    # Load a model
    model = YOLO("./models/yolo11n-cards-60-epochs.pt")

    # Evaluate the models
    results = model.val(data=dataset, conf=0.25,
                        iou=0.65, save_json=True, plots=True, split="test")


# I have to do this since this calls a GPU process
if __name__ == '__main__':
    main()
