import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

yolo_model = YOLO("yolov8n-face.pt")
mask_model = load_model("../models/face_mask_model.keras")

labels = ["No Mask", "Mask"]

def detect_and_classify(frame):
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