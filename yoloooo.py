import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np
import tempfile
import pathlib

# Page config
st.set_page_config(page_title="Fruit Freshness Detector", page_icon="🍎", layout="centered")
st.title("Fruit Freshness Detector 🍌")
st.markdown("Detect if a fruit is **fresh** or **rotten** using AI.")

# Pathlib patch (for Windows)
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLO model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'c:\Users\atray\OneDrive\Documents\Final Project\best.pt', force_reload=True)
    return model

model = load_model()

# Choose input type
option = st.radio("Choose Input Type", ["Image", "Video"])

if option == "Image":
    uploaded_img = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        results = model(img)
        annotated = results.render()[0]

        st.image(annotated, caption="Detected Image", channels="BGR", use_column_width=True)

elif option == "Video":
    uploaded_vid = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
    if uploaded_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())

        vid = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break

            results = model(frame)
            annotated = results.render()[0]

            stframe.image(annotated, channels="BGR", use_column_width=True)
