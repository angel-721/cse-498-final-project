from ultralytics import YOLO
from train import setup_dataset_path
import matplotlib.pyplot as plt


from infer import video_main



# I have to do this since this calls a GPU process
if __name__ == '__main__':
    video_main("./test.mp4")
