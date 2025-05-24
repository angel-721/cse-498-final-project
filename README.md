#  yolo-blackjack-hand-detection

<p align="center">
  <img src="https://github.com/user-attachments/assets/758a64fd-172a-4a1a-8dd6-0ca2d84ffc7d" alt="image" />
</p>

A little project using YOLOv11 to detect BlackJack cards and using a euclidean distance nearest-neighbor method to group detected cards into BlackJack hands.

Gives feedback based off the count of the hand. Currently, it can only detect hands of 2 cards.

TIP: Run scripts from home dir, not src
e.g python src/train.py

READ: Coursesite version will have the demo images provided with the zip, but
not the dataset or models. To train or evaluate the models, please download and
follow the instructions later for setting up the dataset.

## Conda set up

Follow steps below:

```bash

# Create the conda environment
conda env create -f yolo-env.yml

# Activate the conda environment
conda activate yolo-env
```

## Scripts

### `train.py`: Train a YOLOv11 model on cards dataset(TAKES A LONG TIME).

This will start a new training run for 60 epochs under the default settings.

There are helper functions since I ran this on a conda windows environment, so
some local paths didn't work. The train method will attempt to default to using
the aboslute path to the YOLO directory created in `$USER\dataset` but
I wanted to use what was within the project.

### `fine_tune.py`: Fine-tune a YOLOv11 model on cards dataset(TAKES A LONG TIME).

Load the trained model `models/yolo11n-cards-60-epochs.pt` and fine-tune it on
the cards dataset on it. The learning space is small with only looking at the
learning rate of (0.1, 0.00001). It will tune for 20 epochs and do this 15

Results will be saved to `$USER\runs\detect\tune(n)`
(n) is whatever the run number is, so it will be saved to the same directory as
the 5 times if it's your fifth time running the program.

Plots will be saved to keep track of the validation loss and mAP score for each
tune training epoch and the best model will be saved to the same directory.

You may need to modify the data.yaml file to point to the correct location of
the validation set. A line that reads `val: ./valid/images` should be there.

### `test.py`: Test a YOLOv11 model on cards dataset.

This will test the model on the test set and print out the mAP score and mean
recall in addition to hyperparameters used to train the model. This will also
make a directory called val and store plots and results of the testing run.

Before running, ensure that you have downloaded a model and that it's the
parameter to the main function. Default is `./models/tuned-2.pt`.

### `eval.py`: Mainly used to evaluate the first model.

Only used if you want to run the validation set on the first model. Works almost
the same as `test.py`, but will run on the validation set instead of the test.

Default model is `./models/yolo11n-cards-60-epochs.pt`

You may need to modify the data.yaml file to point to the correct location of
the validation set. A line that reads `val: ./valid/images` should be there.

### `infer.py`: Inference on a YOLOv11 model on cards dataset.

This is where all of the business logic is contained. Helper methods to help
calculate centers of bounding boxes, euclidean distance between two points,
finding nearest neighbors for a card, generating hands, and drawing the business
logic feedback on the frame.

Comment out the specific form of inference you want to run. All of them will use
the same business logic and model, but changes the input source of image fed
into the model(webcam, video, or image).

Current setup is to find the nearest neighbors of a card that's at least less
than 180 pixels in euclidean distance away from the card, this worked better on
the demo when generating hands.

- webcam
- video
- image

Note: there is no demo.mp4 provided, this is just a placeholder for a video
file.

This will default to the webcam, but you can change the source to a video or
image by uncommenting the specific line at the bottom.

Default model is `./models/tuned-2.pt`.

### `demo.py`: A specific demo to show off the project.

Only run if you have a video source connected to your computer.

The model of this relies on whatever the model is in `infer.py`.

This will display the 2 demo pictures then the webcam feed. Every frame will be
ran through the model and feedback will be drawn on the frame. Make sure to
press q or esc to each CV2 window to close it.

## Running any of the program

Since the GPU should be utilized when running any of these scripts, I had to
wrap the main functionality into a main function that's guarded by a `if
__name__ == '__main__':` statement due to the parallel nature of calling the GPU
in Python.

DOWNLOAD the [dataset](https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset/data)
from Kaggle.

You may need to modify the data.yaml file to point to the correct location of
the validation set. A line that reads `val: ./valid/images` should be there.

Unzip the dataset and place it in the folder `datasets/cards`.

Ensure you are in the conda environment `yolo-env` and run the script you want to run.

Make sure you have a model before running `infer.py`, `demo.py`, `fine_tune.py`,
or `test.py`.

A zip of different models can be [here](https://drive.google.com/file/d/1EydDJyGpJNLxtZ-nlK5gRjuhAhkWl83d/view?usp=sharing)
Unzip the models and place them in the folder `models`.

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

