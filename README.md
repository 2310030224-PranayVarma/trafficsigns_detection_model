---

# Traffic Sign Detection Using CNN

This project implements a traffic sign detection system using a Convolutional Neural Network (CNN) and the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The system is capable of detecting traffic signs from images and classifying them based on trained models.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Testing with GUI](#testing-with-gui)
- [File Structure](#file-structure)
- [Results](#results)
- [Credits](#credits)

## Overview
This project trains a CNN to classify traffic signs using the GTSRB dataset. The model achieves high accuracy on the test set and can be used to classify traffic signs from user-uploaded images through a simple GUI.

## Dataset
The dataset used is the [German Traffic Sign Recognition Benchmark (GTSRB)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?resource=download), which contains over 50,000 images of 43 different types of traffic signs. 

The dataset contains:
- **Meta folder**: Metadata related to the dataset.
- **Test folder**: Images for testing.
- **Train folder**: Images for training, each subfolder represents a class.

## Model Architecture
The Convolutional Neural Network (CNN) used for traffic sign detection consists of:
- **Convolutional Layers**: For extracting features from the images.
- **MaxPooling Layers**: For downsampling.
- **Dense Layers**: Fully connected layers to make predictions.
- **Activation**: ReLU activation for hidden layers and softmax for output layer.
  
The model is trained with the Adam optimizer and categorical cross-entropy loss.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/traffic_sign_detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd traffic_sign_detection
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

1. Download and extract the [GTSRB dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?resource=download) into the `data` directory of the project. Ensure the structure is as follows:
    ```
    data/
      GTSRB/
        Train/
        Test/
        Meta/
    ```

2. Train the model:
    ```bash
    python src/train.py
    ```

The trained model will be saved as `traffic_sign_model.h5` in the `model/` directory.

## Traffic Sign Labels

The following labels are used for supervised machine learning. Each label corresponds to a specific traffic sign in the dataset:
_____________________________________________
| Label | Description                       |
|-------|-----------------------------------|
| 0     | Speed limit (20km/h)              |
| 1     | Speed limit (30km/h)              |
| 2     | Speed limit (50km/h)              |
| 3     | Speed limit (60km/h)              |
| 4     | Speed limit (70km/h)              |
| 5     | Speed limit (80km/h)              |
| 6     | End of speed limit (80km/h)       |
| 7     | Speed limit (100km/h)             |
| 8     | Speed limit (120km/h)             |
| 9     | No passing                        |
| 10    | No passing veh over 3.5 tons      |
| 11    | Right-of-way at intersection      |
| 12    | Priority road                     |
| 13    | Yield                             |
| 14    | Stop                              |
| 15    | No vehicles                       |
| 16    | Veh > 3.5 tons prohibited         |
| 17    | No entry                          |
| 18    | General caution                   |
| 19    | Dangerous curve left              |
| 20    | Dangerous curve right             |
| 21    | Double curve                      |
| 22    | Bumpy road                        |
| 23    | Slippery road                     |
| 24    | Road narrows on the right         |
| 25    | Road work                         |
| 26    | Traffic signals                   |
| 27    | Pedestrians                       |
| 28    | Children crossing                 |
| 29    | Bicycles crossing                 |
| 30    | Beware of ice/snow                |
| 31    | Wild animals crossing             |
| 32    | End speed + passing limits        |
| 33    | Turn right ahead                  |
| 34    | Turn left ahead                   |
| 35    | Ahead only                        |
| 36    | Go straight or right              |
| 37    | Go straight or left               |
| 38    | Keep right                        |
| 39    | Keep left                         |
| 40    | Roundabout mandatory              |
| 41    | End of no passing                 |
| 42    | End no passing veh > 3.5 tons     |
---------------------------------------------

This table clearly lists each traffic sign label along with its corresponding description. Only these Traffics signs are trained as mentioned by supervised learning

### Testing with GUI

1. After training the model, you can test it by running the GUI for image testing.

2. Run the GUI:
    ```bash
    python src/gui.py
    ```

3. In the GUI, click "Upload Image" to select an image of a traffic sign. The model will classify the image and display the predicted traffic sign.

## File Structure
```
traffic_sign_detection/
│
├── data/
│   └── GTSRB/
│       ├── Train/        # Training images
│       ├── Test/         # Testing images
│       └── Meta/         # Metadata
│
├── model/
│   └── traffic_sign_model.h5  # Saved model after training
│
├── src/
│   ├── train.py          # Code for training the model
│   ├── gui.py            # GUI code for testing the model
│   ├── data_loader.py    # Data loading and preprocessing functions
│   └── traffic_sign_model.py # Model architecture definition
│
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## Results
After training, the model achieves a test accuracy of approximately **99.2%** on the GTSRB dataset. This high accuracy demonstrates the model's ability to correctly classify traffic signs.

## Credits
- Dataset: [German Traffic Sign Recognition Benchmark (GTSRB)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?resource=download)
- TensorFlow and Keras for model building.
- OpenCV for image preprocessing.

---

This `README.md` provides a clear, structured overview of the project, guiding the user on how to set up, train, and test the model.
