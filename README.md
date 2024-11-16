# 📘 American Sign Language Detection and Translation System

## 📝 Project Description
The **American Sign Language Detection and Translation System** is an AI-powered project designed to translate American Sign Language (ASL) gestures into text in real-time using computer vision and deep learning. The system leverages a Convolutional Neural Network (CNN) model trained on a dataset of ASL gestures to accurately recognize and translate signs into readable text.

The goal of this project is to bridge the communication gap for the deaf and hard-of-hearing community by providing a real-time translation tool that can facilitate better interactions.

---

## 📂 Table of Contents
- Project Features
- Tech Stack
- Installation
- Usage
- Dataset
- Model Architecture
- Future Enhancements
- Contributing
- License

---

## 🚀 Project Features
- **Real-time ASL Detection**: Captures video from a webcam and detects ASL gestures.
- **Text Translation**: Converts detected gestures into corresponding text.
- **Scalable and Extendable**: Built using Python and TensorFlow, allowing easy modifications and upgrades.
- **Pre-trained Model**: Includes a pre-trained CNN model for efficient ASL recognition.

---

## 💻 Tech Stack
- **Languages**: Python
- **Libraries/Frameworks**: 
  - TensorFlow/Keras
  - OpenCV
  - MediaPipe (for hand detection)
  - NumPy
  - Matplotlib

---

## ⚙️ Installation

To get started, follow these instructions to set up the project on your local machine.

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- `pip` package manager

### Clone the Repository
git clone https://github.com/idaraabasiudoh/American-Sign-Language-ASL-Detection.git 
cd ASL-Detection-System

### Install Dependencies
pip install tensorflow opencv-python opencv-python-headless mediapipe numpy matplotlib

---

## 🛠 Usage

### Step 1: Preprocess the Dataset
python data_preprocessing.py

### Step 2: Train the Model
python model_training.py

### Step 3: Real-Time ASL Detection
To run the real-time detection using your webcam:
python real_time_detection.py

The system will open a video window where you can perform ASL gestures in front of the camera, and the corresponding letter will be displayed on the screen.

---

## 📊 Dataset
The model is trained using the [ASL Alphabet Dataset](https://www.kaggle.com/grassknoted/asl-alphabet), which includes images of hands representing the 26 letters of the English alphabet. The dataset has been preprocessed and split into training, validation, and test sets.

### Data Preprocessing Steps
- Grayscale conversion
- Image resizing to 64x64 pixels
- Normalization of pixel values to [0, 1]
- Data augmentation for robust model training

---

## 🧠 Model Architecture
The model is a Convolutional Neural Network (CNN) with the following layers:
- **Conv2D + MaxPooling**: Feature extraction layers.
- **Flatten**: Converts the 2D feature maps into a 1D feature vector.
- **Dense**: Fully connected layers for classification.
- **Dropout**: Prevents overfitting.

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])
```

---

## 🤝 Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

---

## 📝 License
This project is licensed under the MIT License - see the `LICENSE` file for details.

---
