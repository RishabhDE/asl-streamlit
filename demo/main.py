import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch import nn
import streamlit_webrtc as webrtc
import cv2
from utils import load_and_preprocess_image, predict_image
from efficientnet_pytorch import EfficientNet

from torchvision import models


def load_model():
    MODEL_LOAD_PATH = "/mount/src/asl-streamlit/demo/efficientnet_model.pth"

    # Initialize the model as in training
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_classes = 29  # Adjust to your number of classes
    model._fc = nn.Linear(model._fc.in_features, num_classes)

    # Load the state_dict
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Real-time webcam feed callback
class VideoProcessor:
    def __init__(self):
        self.model = load_model()  # Load model once
        self.class_names = np.array([chr(i) for i in range(ord('A'), ord('Z')+1)] + ['del', 'nothing', 'space'])
        self.detected_letters = []

    def recv(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to RGB for PIL Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Preprocess and predict the image
        transformed_image = load_and_preprocess_image(img_pil)  # Ensure this function is adapted for your model input
        predicted_label, _ = predict_image(self.model, transformed_image, self.class_names)

        # If prediction is 'del', remove the last letter from detected_letters
        if predicted_label == 'del':
            if self.detected_letters:
                self.detected_letters.pop()

        elif predicted_label != 'nothing' and predicted_label != 'space':
            # If detected letter is valid, append it to the sequence
            self.detected_letters.append(predicted_label)

        # Update the sequence display
        st.write("Detected Letters: " + "".join(self.detected_letters))

        # Return the frame to be displayed
        return frame

# Streamlit interface
def main():
    st.title("Real-time ASL Sign Language Recognition")

    # Initialize webcam with Streamlit WebRTC
    webrtc_config = webrtc.WebRtcMode.SENDRECV
    video_processor = VideoProcessor()

    # Start the webcam feed with video processing callback
    st_webrtc = webrtc.streamlit_webrtc(
        video_processor=video_processor,
        mode=webrtc_config,
        video_frame_callback=video_processor.recv,
    )

if __name__ == "__main__":
    main()
