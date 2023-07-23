import os
import tensorflow as tf
from PIL import Image
import numpy as np
import tensorflow as tf


def extract_rgb_info_from_images(folder_path):
    image_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Initialize an empty list to store RGB information
    rgb_info_list = []

    # Loop through each image in the folder
    for image_path in image_paths:
        # Open the image using PIL
        image = Image.open(image_path)

        # Convert the image to RGB mode (in case it's grayscale or other modes)
        image_rgb = image.convert("RGB")

        # Convert the image to a NumPy array
        image_array = np.array(image_rgb)

        # Append the RGB array to the list
        rgb_info_list.append(image_array)

    # Convert the list of RGB arrays to a NumPy array
    rgb_info_array = np.array(rgb_info_list)

    return rgb_info_array


folder_path_0 = "_0-patch"
rgb_info_array_0 = extract_rgb_info_from_images(folder_path_0)

folder_path_N1 = "_N1-patch"
rgb_info_array_N1 = extract_rgb_info_from_images(folder_path_N1)

folder_path_valX = "_valX-patch"
rgb_info_array_valX = extract_rgb_info_from_images(folder_path_valX)

folder_path_valY = "_valY-patch"
rgb_info_array_valY = extract_rgb_info_from_images(folder_path_valY)

# Print the shape of the RGB information array
print("Shape of RGB information array:", rgb_info_array_0.shape)
print("Shape of RGB information array:", rgb_info_array_N1.shape)
print("Shape of RGB information array:", rgb_info_array_valX.shape)
print("Shape of RGB information array:", rgb_info_array_valY.shape)

# model


def create_autoencoder_model(input_shape):
    model = tf.keras.Sequential()

    # Encoder
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))

    # Decoder
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(tf.keras.layers.UpSampling2D((2, 2)))
    model.add(
        tf.keras.layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")
    )  # Output layer

    return model


trainX = rgb_info_array_0
trainY = rgb_info_array_N1
valX = rgb_info_array_valX
valY = rgb_info_array_valY


# Normalize pixel values to [0, 1]
trainX = trainX.astype("float32") / 255.0
trainY = trainY.astype("float32") / 255.0
valX = valX.astype("float32") / 255.0
valY = valY.astype("float32") / 255.0

# Create the autoencoder model
input_shape = trainX.shape[1:]
model = create_autoencoder_model(input_shape)

# Compile the model
model.compile(optimizer="adam", loss="mse")

# Train the model
model.fit(trainX, trainY, batch_size=64, epochs=10, validation_data=(valX, valY))

# Evaluate the model on validation data
loss = model.evaluate(valX, valY)
print("Validation loss:", loss)

# Save the model for later use
model.save("autoencoder_model.h5")
