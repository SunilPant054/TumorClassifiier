from data.data_loader import *
from model.network import ClassifierModel
from model.predict import Predict
from model.train_network import TrainNetwork
from view.ui import Frontend


class Main:
    base_dir = "/home/pneuma/Desktop/ML/Deep Learning/TumorClassification/dataset/"

    # load data
    def __init__(self):
        data = DataLoader(self.base_dir)
        self.train, self.test, self.val = data.load_data()


        # compile model
        m1 = ClassifierModel()
        self.model = m1.network()

        # Fit Model
        fit = TrainNetwork(self.train, self.val)
        fit.train_model(self.model)

        # Predict
        self.predict = Predict(self.test)

    # ui
        self.view = Frontend()
        self.view.model = self.model
        self.view.ui(self.predict.predict_model)

if __name__ == "__main__":
    Main()