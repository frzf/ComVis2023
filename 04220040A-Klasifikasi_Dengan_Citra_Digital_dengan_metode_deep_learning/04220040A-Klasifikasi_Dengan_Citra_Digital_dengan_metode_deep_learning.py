# Import library yang dibutuhkan
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Membuat model deep learning
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, img_channels)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Membuat data train dan data validation
train_data = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_data.flow_from_directory(directory=train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode="categorical", subset="training")
validation_generator = train_data.flow_from_directory(directory=train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode="categorical", subset="validation")

# Melatih model
model.fit(train_generator, epochs=num_epochs, validation_data=validation_generator)

# Menggunakan model untuk klasifikasi citra
test_data = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_data.flow_from_directory(directory=test_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode="categorical")
model.evaluate(test_generator)