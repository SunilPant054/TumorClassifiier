from data.data_loader import *


class main:
    base_dir = "/home/pneuma/Desktop/ML/Deep Learning/TumorClassification/dataset/"

    # load data
    data = DataLoader(base_dir)
    train, test, val = data.load_data()