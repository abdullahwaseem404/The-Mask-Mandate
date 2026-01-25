# 🧑‍💻 The-Mask-Mandate
# Face Mask Detection using TensorFlow & MobileNetV2

This project implements a **binary image classification model** to detect whether a person is wearing a face mask or not. It leverages **TensorFlow** with the **MobileNetV2** architecture for transfer learning, combined with **OpenCV** and **Streamlit** for deployment and visualization.

---

## 🚀 Features
- Pretrained **MobileNetV2** backbone for efficient transfer learning.
- Image preprocessing with **ImageDataGenerator** (train/validation split).
- Binary classification (`mask` vs `no mask`).
- GPU/CPU compatibility check.
- Model training with dropout regularization to prevent overfitting.
- Exported trained model (`face_mask_model.keras`).
- Ready for deployment with **Streamlit**.

---

## 📂 Project Structure
```
├── dataset/                # Dataset folder with 'mask' and 'no_mask' subfolders
├── face_mask_model.keras   # Saved trained model
├── app.py                  # (Optional) Streamlit app for demo
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/abdullahwaseem404/The-Mask-Mandate.git
```
```bash
pip install -r requirements.txt
```


---

## 📊 Training

Run the training script:

```bash
python train.py
```

Key parameters:
- `IMG_SIZE = 224`
- `BATCH_SIZE = 16`
- `EPOCHS = 3`
- `validation_split = 0.2`

The model will be saved as:

```
face_mask_model.keras
```

---

## 🖼️ Dataset

The dataset should be organized as follows:

```
dataset/
│── mask/
│   ├── image1.jpg
│   ├── image2.jpg
│── no_mask/
│   ├── image3.jpg
│   ├── image4.jpg
```

---

## 📈 Model Summary

The model uses **MobileNetV2** (pretrained on ImageNet) as the base, with custom dense layers:

- Flatten
- Dense(128, ReLU)
- Dropout(0.5)
- Dense(1, Sigmoid)

Loss: `binary_crossentropy`  
Optimizer: `adam`  
Metrics: `accuracy`

---

## 🌐 Deployment (Streamlit)

Run the Streamlit app:

```bash
streamlit run app.py
```

This will launch a web interface where users can upload images and get predictions.
