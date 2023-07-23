import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def create_autoencoder_model(input_shape):
    # Same as before
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


def apply_autoencoder_to_image(model, input_image, patch_size):
    height, width, _ = input_image.shape
    edited_image = np.zeros_like(input_image)

    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = input_image[y : y + patch_size, x : x + patch_size]
            patch = np.expand_dims(patch, axis=0)
            patch = patch.astype("float32") / 255.0
            edited_patch = model.predict(patch)
            edited_patch = (edited_patch[0] * 255).astype("uint8")
            edited_image[y : y + patch_size, x : x + patch_size] = edited_patch

    return edited_image


# Load the trained autoencoder model
model = tf.keras.models.load_model("autoencoder_model.h5")

# Load the input image you want to edit
input_image_path = "test_image.JPG"  # Replace with the actual path of the input image
input_image = Image.open(input_image_path)
input_image = np.array(input_image)

# Set the patch size
patch_size = 32

# Apply the autoencoder to the input image patches
edited_image = apply_autoencoder_to_image(model, input_image, patch_size)

# Display the original and edited images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Load and display the original image
axes[0].imshow(input_image)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Display the edited image
axes[1].imshow(edited_image)
axes[1].set_title("Edited Image")
axes[1].axis("off")

plt.show()
