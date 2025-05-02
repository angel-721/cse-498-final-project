from ultralytics import YOLO
from train import setup_dataset_path
import matplotlib.pyplot as plt


def main():
    dataset = setup_dataset_path()

    # Load a model
    model = YOLO("./models/yolo11n-cards-60-epochs.pt")

    # Evaluate the models
    results = model.val(data=dataset, conf=0.25,
                        iou=0.65, save_json=True, plots=True)

    # Print the metrics
    # print(f"Results: {metrics}")
    print("Class indices with average precision:", results.ap_class_index)
    print("Average precision for all classes:", results.box.all_ap)
    print("Mean average precision at IoU=0.50:", results.box.map50)
    print("Mean recall:", results.box.mr)


# I have to do this since this calls a GPU process
if __name__ == '__main__':
    main()
