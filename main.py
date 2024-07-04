from data.data_loader import *
from model.network import ClassifierModel
from model.train_network import TrainNetwork



class main:
    base_dir = "/home/pneuma/Desktop/ML/Deep Learning/TumorClassification/dataset/"

    # load data
    data = DataLoader(base_dir)
    train, test, val = data.load_data()
    
    #compile model
    m1 = ClassifierModel()
    model = m1.network()
    
    #Fit Model 
    fit = TrainNetwork(train, val)
    fit.train_model(model)