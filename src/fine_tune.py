from ultralytics import YOLO
from train import setup_dataset_path


def main():
    # Initialize the YOLO model
    model = YOLO("./models/yolo11n-cards-60-epochs.pt")


    dataset = setup_dataset_path()

    # Define search space
    search_space = {
        "lr0": (1e-5, 1e-1),
    }

    # Tune hyperparameters
    print("Starting tuning...")
    model.tune(
        data=dataset,
        epochs=20,
        iterations=15,
        optimizer="AdamW",
        space=search_space,
        plots=True,
        val=True,
    )

    model.save("./models/yolo11n-cards-tuned.pt")

if __name__ == "__main__":
    main()
