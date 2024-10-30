from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess images
def load_images(folder_path, target_size=(64, 64)):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg', 'jfif'))]
    print(f"Found {len(image_files)} images.")
    images = []
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        image = np.array(image) / 255.0
        images.append(image)
    
    print(f"Loaded {len(images)} images.")
    return np.array(images)

# Define a simple CNN model
def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))  # Assume binary classification for simplicity
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Plot images and predictions
def plot_images_with_predictions(images, predictions):
    plt.figure(figsize=(10, 5))
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)  # Adjusted to match the number of images
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Pred: {np.argmax(predictions[i])}")
    plt.show()

# Folder containing images
folder_path = '/content/img'

# Load and preprocess images
images = load_images(folder_path)

# Create CNN model
input_shape = (64, 64, 3)  # Image shape (64x64 RGB images)
model = create_cnn_model(input_shape)

# Generate labels to match the number of loaded images
num_images = images.shape[0]
labels = np.random.randint(2, size=(num_images, 2))  # Random labels matching the number of images

# Train the model on the loaded images
model.fit(images, labels, epochs=3)

# Get predictions from the model
predictions = model.predict(images)

# Plot the images with their predicted labels
plot_images_with_predictions(images, predictions)
