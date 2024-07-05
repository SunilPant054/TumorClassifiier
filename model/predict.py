import numpy as np
import cv2


class Predict:
    def __init__(self, test) -> None:
        self.test = test

    def predict_model(self, model, image):
        # Convert the image from a NumPy array to an OpenCV format if needed
        if isinstance(image, np.ndarray):
            # Convert RGB to BGR format for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Resize the image to match model input size
        # Resize to match model input size
        image = cv2.resize(image, (256, 256))
        image_array = image / 255.0  # Normalize the image
        image_array = np.expand_dims(
            image_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(x=image_array)
        max_prediction = np.argmax(predictions, axis=1)

        # Convert the prediction to a human-readable label
        class_labels = list(self.test.class_indices.keys())
        predicted_label = class_labels[max_prediction[0]]
        return predicted_label
