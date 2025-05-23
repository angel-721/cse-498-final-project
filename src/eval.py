from ultralytics import YOLO
from train import setup_dataset_path


def main():
    dataset = setup_dataset_path()

    # Load a model
    model = YOLO("./models/yolo11n-cards-60-epochs.pt")
    # model = YOLO("./models/tuned-1.pt")
    # model = YOLO("./models/tuned-2.pt")

    # Evaluate the models
    results = model.val(data=dataset, conf=0.25,
                        iou=0.65, plots=True, split="test")

    # Print the metrics
    print("Map @ IoU 50:95:", results.box.map)
    print("Mean recall:", results.box.mr)

    #print parameters
    print("hyperparameters:", model.model.yaml)


# I have to do this since this calls a GPU process
if __name__ == '__main__':
    main()
