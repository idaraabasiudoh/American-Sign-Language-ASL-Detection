import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Set paths
TRAIN_PATH = 'ASL_Alphabet_Dataset/asl_alphabet_train'
TEST_PATH = 'ASL_Alphabet_Dataset/asl_alphabet_test'
IMG_SIZE = 64

# Create a dictionary to map folder names to numerical labels
LABELS_DICT = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9,
    "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19,
    "U": 20, "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25, "del": 26, "nothing": 27, "space": 28
}

def load_data_from_folder(folder_path):
    """
    Load images and labels from a specified folder path.
    """
    data = []
    labels = []
    
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path) and label in LABELS_DICT:
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    data.append(img)
                    labels.append(LABELS_DICT[label]) 
    
    data = np.array(data) / 255.0  # Normalize images to range [0, 1]
    data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Add channel dimension
    labels = np.array(labels)
    
    return data, labels

def load_dataset():
    """
    Load training and test datasets.
    """
    # Load training data
    train_data, train_labels = load_data_from_folder(TRAIN_PATH)
    
    # Load test data
    test_data, test_labels = load_data_from_folder(TEST_PATH)
    
    return train_data, test_data, train_labels, test_labels

# Load datasets
X_train, X_test, y_train, y_test = load_dataset()

# Save preprocessed data to a compressed .npz file
np.savez_compressed('asl_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print("Data preprocessed and saved successfully!")