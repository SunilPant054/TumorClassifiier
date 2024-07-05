import gradio as gr
import numpy as np
import cv2


class Frontend:
    def __init__(self):
        pass

    def ui(self, predict_func):
        def classify_image(image):
            # Convert image from RGB to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Call the predict_model method with the uploaded image
            # Pass the model and the uploaded image
            result = predict_func(self.model, image)
            return result

        gr.Interface(
            fn=classify_image,
            inputs=gr.Image(),  # Updated to use `gr.Image()` without `shape`
            outputs=gr.Label(),  # Updated to use `gr.Label()` for output
            title="Image Classification",
            description="Upload an image to classify."
        ).launch(True)
