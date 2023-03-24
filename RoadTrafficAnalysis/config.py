from dataclasses import dataclass
from datetime import datetime


@dataclass
class Config:
    MAIN_FOLDER_NAME = "Overspeed"
    SUB_FOLDER_NAME = datetime.now().strftime('%d%m%Y')
    FULL_PATH = rf"{MAIN_FOLDER_NAME}\{SUB_FOLDER_NAME}"
    MAX_AGE = 20
    DISTANCE = 20
    MAX_HISTORY = 30
    MODELS_FOLDER = "models"
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    VIDEOS_FOLDER = "Videos"
    VIDEO = "los_angeles.mp4"
    MODEL_WEIGHTS = "yolov3_training_final.weights"
    MODEL_CONFIG = "yolov3_testing.cfg"
    MODEL_CLASSES = "classes.txt"
    INTERFACE_COLOR = (255, 255, 255)
    PANEL_SIZE = (360, 260)
    PANEL_PADDING = 2
    ALPHA = 0.4
