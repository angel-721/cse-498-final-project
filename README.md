# cse-498-final-project

TIP: Run scripts from home dir, not src
e.g python src/train.py

## Conda set up

Follow steps below:

```bash

# Create the conda environment
conda env create -f yolo-env.yml

# Activate the conda environment
conda activate yolo-env
```

## Scripts

### `train.py`: Train a YOLOv11 model on cards dataset.

### `fine_tune.py`: Fine-tune a YOLOv11 model on cards dataset.

### `test.py`: Test a YOLOv11 model on cards dataset.

### `eval.py`: Mainly used to evaluate the first model.

### `infer.py`: Inference on a YOLOv11 model on cards dataset.

Comment out the specific form of inference you want to run.

- webcam
- video
- image

### `demo.py`: A specific demo to show off the project.

Works like `infer.py` but with a different setup.

## Running any of the program

DOWNLOAD the [dataset](https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset/data)
from Kaggle.

Unzip the dataset and place it in the folder `datasets/cards`.

Ensure you are in the conda environment `yolo-env` and run the script you want to run.

Make sure you have a model before running `infer.py`, `demo.py`, `fine_tune.py`,
or `test.py`.

A trained model can be found [here](put link later zip)

## Resources

### Training

- https://docs.ultralytics.com/models/yolo11/
- https://docs.ultralytics.com/modes/train/#key-features-of-train-mode

### Evaluation/testing

- https://docs.ultralytics.com/guides/model-evaluation-insights/#mean-average-precision

### Inference/Tracking

- https://docs.ultralytics.com/modes/track/
- https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/

### Conda

- https://pytorch.org/get-started/previous-versions/
- https://docs.ultralytics.com/guides/conda-quickstart/#setting-up-a-conda-environment

### Dataset

- https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset/data

