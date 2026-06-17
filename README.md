# 😷 The-Mask-Mandate – Face Mask Detection System

A real-time **Face Mask Detection System** that uses YOLOv8 for face detection and a deep learning classifier (MobileNetV2) to determine whether a person is wearing a mask or not.

---

## 🚀 Features

* 📷 Real-time webcam detection
* 🧠 Dual-model pipeline:
  * YOLOv8 for face detection
  * MobileNetV2 for mask classification
* 🎯 Bounding box + label visualization
* ⚡ Fast and lightweight inference
* 🌐 Streamlit-based interactive UI

---

## 🧠 Models Used
### 🔹 YOLOv8 (Ultralytics)
* Detects faces in real-time
* High speed and accuracy
### 🔹 MobileNetV2 (TensorFlow/Keras)
* Pretrained on ImageNet
* Fine-tuned for mask classification

---

## ⚙️ Installation

```bash id="mask2"
git clone https://github.com/abdullahwaseem404/The-Mask-Mandate.git
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Streamlit app:

```bash id="mask4"
streamlit run app.py
```

---

## 🎯 How It Works

1. 📸 Capture frame from webcam
2. 👤 YOLOv8 detects faces
3. ✂️ Crop detected face region
4. 🧠 MobileNetV2 predicts mask/no-mask
5. 🟢 Draw bounding box with label

---

## 🧪 Model Training

* Uses data augmentation
* Transfer learning with MobileNetV2
* Binary classification (Mask / No Mask)

---

## 📊 Output

* 🟢 Green box → Mask
* 🔴 Red box → No Mask

---
