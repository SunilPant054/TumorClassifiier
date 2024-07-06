from data.data_loader import *
from model.network import ClassifierModel
from model.predict import Predict
from model.train_network import TrainNetwork
from view.ui import Frontend
import tensorflow as tf
from keras._tf_keras.keras.models import load_model


class Main:
    base_dir = "/home/pneuma/Desktop/ML/Deep Learning/TumorClassification/dataset/"
    model_path = "/home/pneuma/Desktop/ML/Deep Learning/TumorClassification/model/model.h5"
    # load data

    def __init__(self):
        data = DataLoader(self.base_dir)
        self.train, self.test, self.val = data.load_data()

        # Check if the model exists
        try:
            self.model = load_model(self.model_path)  # Load the saved model
        except:
            # compile model
            m1 = ClassifierModel()
            self.model = m1.network()

            # Fit Model
            fit = TrainNetwork(self.train, self.val)
            self.model, self.history = fit.train_model(self.model)

            # Save the trained model
            self.model.save(self.model_path)

        # Predict
        self.predict = Predict(self.test)
        self.predict.model = self.model
        test_accuracy, sample_predictions = self.predict.predict_model(self.model)
        print(test_accuracy, sample_predictions)

        # #ui
        # self.view = Frontend()
        # self.view.ui(self.predict.predict_single_image)
        self.view = Frontend()
        self.view.set_model(self.model)  # Set the model for the Frontend
        self.view.ui()


if __name__ == "__main__":
    Main()
