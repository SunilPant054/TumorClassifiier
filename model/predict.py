import numpy as np
import cv2


class Predict:
    def __init__(self, test) -> None:
        self.test = test
        # self.model = model

    def predict_model(self, model):
        """
        Predict using the data and return the accuracy and sampled prediction
        """
        # Calculate accuracy using the model's evaluate method
        test_loss, test_accuracy, = model.evaluate(self.test, verbose=2)

        # Prepare sample predictions
        sample_predictions = []
        for img_batch, labels in self.test:
            predictions = model.predict(img_batch)
            for i in range(len(labels)):
                predicted_class = np.argmax(predictions[i])
                confidence = np.max(predictions[i]) * 100

                sample_predictions.append({
                    'True Label': np.argmax(labels[i]),
                    'Predicted Label': predicted_class,
                    'Confidence': confidence
                })
            break
        return test_accuracy, sample_predictions
