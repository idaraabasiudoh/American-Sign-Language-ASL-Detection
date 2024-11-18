# American Sign Language (ASL) Recognition System

This repository contains the code and resources for an **American Sign Language (ASL) Recognition System**, which leverages machine learning to recognize ASL gestures in real-time. The project consists of data preprocessing, model training, and deployment for real-time ASL recognition via a graphical user interface.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Overview

The ASL Recognition System uses deep learning to identify and classify ASL gestures from real-time video input. The model supports letters A-Z and common actions like "delete," "nothing," and "space." The solution provides a complete pipeline, from data preprocessing to model training and deployment, making it easy to build and extend.

## Features

- **Data Preprocessing**: Efficient scripts to preprocess video/image data for model training.
- **Model Training**: Robust training with a TensorFlow-based model saved in `asl_model.h5`.
- **Real-Time Detection**: Real-time gesture recognition using a GUI-based system.
- **Extensibility**: Modular structure for easy integration and updates.

## File Structure

```plaintext
├── asl_model.h5                # Trained model
├── labels.txt                  # Labels for ASL gestures
├── data_preprocessing.py       # Script for data preprocessing
├── model_transform.py          # Helper functions for model transformations
├── model_training.py           # Model training script
├── test.py                     # Script for testing the model
├── real_time_detection.py      # Real-time detection implementation
├── model_GUI.py                # GUI for gesture recognition
├── LICENSE                     # License information
```

## Getting Started

### Prerequisites
- Python 3.7 or higher
- TensorFlow 2.x
- OpenCV
- Tkinter (for GUI)
- Other Python dependencies (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/idaraabasiudoh/asl-recognition-system.git
   cd asl-recognition-system
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download any necessary datasets for training and preprocessing.

## Usage

### 1. Preprocess Data
Run the preprocessing script to prepare your dataset:
```bash
python data_preprocessing.py
```

### 2. Train the Model
Train the ASL recognition model using:
```bash
python model_training.py
```

### 3. Test the Model
Evaluate the model performance:
```bash
python test.py
```

### 4. Real-Time Recognition
Launch the GUI for real-time gesture detection:
```bash
python model_GUI.py
```

## Labels

The `labels.txt` file contains the gesture classes:
```
A, B, C, ..., Z, delete, nothing, space
```

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request with your enhancements.

## Acknowledgments

- TensorFlow and OpenCV for their powerful frameworks.
- ASL communities for inspiration and datasets.
- Open-source contributors for maintaining useful libraries.

**Author**: [idaraabasiudoh](https://github.com/idaraabasiudoh)
