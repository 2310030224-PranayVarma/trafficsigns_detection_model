import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow import keras
from PIL import Image, ImageTk  # Import Pillow for image handling

class TrafficSignModel:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.image_size = (64, 64)  # Set to the size used during training

    def predict(self, image_path):
        # Load and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.image_size)
        image = np.expand_dims(image, axis=0) / 255.0  # Normalize the image
        predictions = self.model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        return predicted_class  # Return the predicted class index

class TrafficSignApp:
    def __init__(self, master):
        self.master = master
        self.model = TrafficSignModel('model/traffic_sign_model.h5')  # Adjust path to your saved model
        self.master.title("Traffic Sign Detection")
        self.master.geometry("600x600")  # Increase the window size

        # Title Label
        self.title_label = tk.Label(master, text="Traffic Sign Detection", font=("Helvetica", 20))
        self.title_label.pack(pady=20)

        # Create a button to upload images
        self.upload_button = tk.Button(master, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        # Label to display the uploaded image
        self.image_label = tk.Label(master)
        self.image_label.pack(pady=10)

        # Create a label to display the result
        self.result_label = tk.Label(master, text="", font=("Helvetica", 16))
        self.result_label.pack(pady=20)

        # Create a button for detection
        self.detect_button = tk.Button(master, text="Detect", command=self.detect_sign, bg="black", fg="white")
        self.detect_button.pack(pady=10)
        self.detect_button.config(state=tk.DISABLED)  # Disable initially

        self.file_path = None  # Store the uploaded image path

    def upload_image(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            try:
                # Display the uploaded image
                uploaded_image = Image.open(self.file_path)
                uploaded_image = uploaded_image.resize((200, 200))  # Resize for display
                self.image_label.image = ImageTk.PhotoImage(uploaded_image)
                self.image_label.config(image=self.image_label.image)

                self.result_label.config(text="")  # Clear previous results
                self.detect_button.config(state=tk.NORMAL)  # Enable the detect button
            except Exception as e:
                messagebox.showerror("Error", f"Error in loading image: {str(e)}")

    def detect_sign(self):
        if self.file_path:
            try:
                predicted_class = self.model.predict(self.file_path)
                label = self.get_label(predicted_class)
                self.result_label.config(text=f"Predicted Traffic Sign: {label}")
            except Exception as e:
                messagebox.showerror("Error", f"Error in prediction: {str(e)}")

    def get_label(self, class_index):
        # Update this mapping according to your dataset classes
        labels = {
            0: 'Speed limit (20km/h)',
            1: 'Speed limit (30km/h)',
            2: 'Speed limit (50km/h)',
            3: 'Speed limit (60km/h)',
            4: 'Speed limit (70km/h)',
            5: 'Speed limit (80km/h)',
            6: 'End of speed limit (80km/h)',
            7: 'Speed limit (100km/h)',
            8: 'Speed limit (120km/h)',
            9: 'No passing',
            10: 'No passing veh over 3.5 tons',
            11: 'Right-of-way at intersection',
            12: 'Priority road',
            13: 'Yield',
            14: 'Stop',
            15: 'No vehicles',
            16: 'Veh > 3.5 tons prohibited',
            17: 'No entry',
            18: 'General caution',
            19: 'Dangerous curve left',
            20: 'Dangerous curve right',
            21: 'Double curve',
            22: 'Bumpy road',
            23: 'Slippery road',
            24: 'Road narrows on the right',
            25: 'Road work',
            26: 'Traffic signals',
            27: 'Pedestrians',
            28: 'Children crossing',
            29: 'Bicycles crossing',
            30: 'Beware of ice/snow',
            31: 'Wild animals crossing',
            32: 'End speed + passing limits',
            33: 'Turn right ahead',
            34: 'Turn left ahead',
            35: 'Ahead only',
            36: 'Go straight or right',
            37: 'Go straight or left',
            38: 'Keep right',
            39: 'Keep left',
            40: 'Roundabout mandatory',
            41: 'End of no passing',
            42: 'End no passing veh > 3.5 tons'
        }
        return labels.get(class_index, "Unknown Sign")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()
