from data.data_loader import *
from model.network import ClassifierModel


class main:
    base_dir = "/home/pneuma/Desktop/ML/Deep Learning/TumorClassification/dataset/"

    # load data
    data = DataLoader(base_dir)
    train, test, val = data.load_data()
    
    #compile model
    model = ClassifierModel()
    model.network()