import streamlit as st
import cv2
import numpy as np
from tensorflow import keras

st.set_page_config(page_title="Face Mask Detection", layout="wide")
st.title("😷 Face Mask Detection - Live Camera")

model = keras.models.load_model("face_mask_model.keras")

labels = ["No Mask", "Mask"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if "camera" not in st.session_state:
    st.session_state.camera = cv2.VideoCapture(0)

FRAME_WINDOW = st.image([])

run = st.checkbox("Run Camera", value=True)

while run:
    ret, frame = st.session_state.camera.read()
    if not ret:
        st.error("Failed to access camera.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue

        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face, verbose=0)[0][0]
        label = "Mask" if pred > 0.5 else "No Mask"

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
