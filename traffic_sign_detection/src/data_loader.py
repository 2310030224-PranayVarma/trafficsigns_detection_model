import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    images = []
    labels = []
    
    train_dir = os.path.join(data_dir, 'Train')  # Adjusted to look into the 'Train' folder
    for sign in os.listdir(train_dir):
        sign_path = os.path.join(train_dir, sign)
        
        # Only process directories (folders containing images)
        if os.path.isdir(sign_path):
            for image_file in os.listdir(sign_path):
                image_path = os.path.join(sign_path, image_file)
                print(f"Loading image: {image_path}")  # Debug statement
                image = cv2.imread(image_path)

                # Check if the image was successfully loaded
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                image = cv2.resize(image, (64, 64))  # Resize images
                images.append(image)
                labels.append(int(sign))  # Use the folder name as the label

    return np.array(images), np.array(labels)

def preprocess_data(data_dir):
    images, labels = load_data(data_dir)
    images = images / 255.0  # Normalize pixel values
    return train_test_split(images, labels, test_size=0.2, random_state=42)

if __name__ == "__main__":
    data_dir = 'data/GTSRB/'
    X_train, X_test, y_train, y_test = preprocess_data(data_dir)
    print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

