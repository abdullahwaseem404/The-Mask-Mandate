import streamlit as st
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from inference import detect_and_classify

st.set_page_config(page_title="Face Mask Detection", layout="wide")
st.title("😷 The-Mask-Mandate – Live Detection System")

if "camera" not in st.session_state:
    st.session_state.camera = cv2.VideoCapture(0)

FRAME_WINDOW = st.image([])

run = st.checkbox("Start Camera", value=True)

while run:
    ret, frame = st.session_state.camera.read()

    if not ret:
        st.error("Camera not accessible")
        break

    frame = cv2.flip(frame, 1)

    frame = detect_and_classify(frame)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

st.session_state.camera.release()