import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import preprocess_data
from traffic_sign_model import create_model

def plot_training_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plots as an image
    plt.savefig('training_history.png')

    # Show the plots
    plt.show()

def main():
    data_dir = '/workspaces/trafficsigns_detection_model/traffic_sign_detection/data/GTSRB'

    X_train, X_test, y_train, y_test = preprocess_data(data_dir)

    # Determine the number of classes
    num_classes = len(np.unique(y_train))

    model = create_model(num_classes)

    # Train the model and capture the history
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the model
    model_dir = '/workspaces/trafficsigns_detection_model/traffic_sign_detection/model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model.save(os.path.join(model_dir, 'traffic_sign_model.h5'))
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')

    # Plot the training history
    plot_training_history(history)

if __name__ == "__main__":
    main()
