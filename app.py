import os
import sys
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

st.set_page_config(page_title="Face Mask Detection", layout="wide")
st.title("😷 The-Mask-Mandate – Live Detection System")

yolo_model = YOLO("yolov8n-face.pt")
MODEL_WEIGHTS_PATH = "face_mask_weights.weights.h5"

@st.cache_resource
def load_mask_classifier():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_weights(MODEL_WEIGHTS_PATH)
    return model

mask_model = load_mask_classifier()
labels = ["No Mask", "Mask"]

def detect_and_classify(frame):
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        cv2.putText(frame, "Model weights not found! Run training first.", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    results = yolo_model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_resized = cv2.resize(face, (224, 224)) / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        pred = mask_model.predict(face_resized, verbose=0)[0][0]
        label = "Mask" if pred > 0.5 else "No Mask"

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame

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

if not os.path.exists(MODEL_WEIGHTS_PATH) and os.path.exists("dataset/"):
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2
    )

    train_data = train_datagen.flow_from_directory(
        "dataset/",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training"
    )

    val_data = train_datagen.flow_from_directory(
        "dataset/",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation"
    )

    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )

    model.save_weights(MODEL_WEIGHTS_PATH)