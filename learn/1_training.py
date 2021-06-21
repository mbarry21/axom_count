import properties
import data.orig.orig_data

import pandas as pd
import matplotlib.pyplot as plt

# from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


# Create a Convolutional Neural network with 3 Conv layers and 1 dense layer using a "funnel" layout
def create_model():
    model = tf.keras.Sequential([
        # 128 x 128
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # 64 x 64
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # 32 x 32
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=(2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # 16 x 16
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16 * 16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    return model

# Create a data generator that can modify training data
#  TODO: Data is currently loaded directly from directory and image augmentation is done on the fly. Consider saving
#   augmented data in parquet format segmented by batches
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.2, 1.8),
    validation_split=0.2
)

# TODO: Consider adding the following improvements:
#  -If unlabeled data is more available, consider using an autoencoder as an initial step to simplify learning.
#  -Imbalanced data: Consider applying SMOTE/Weights/sampling/Variational autoencoders if imbalances exist.



#  Read in labels for data as DataFrame
train_labels = pd.DataFrame(data.orig.orig_data.vile_counts)

# Generate train and validation sets
train_generator = datagen.flow_from_dataframe(dataframe=train_labels, directory=properties.resized_path,
                                              x_col="image_name", y_col="count", has_ext=True,
                                              class_mode="other", target_size=(128, 128),
                                              batch_size=1, subset="training")

valid_generator = datagen.flow_from_dataframe(dataframe=train_labels, directory=properties.resized_path,
                                              x_col="image_name", y_col="count", has_ext=True,
                                              class_mode="other", target_size=(128, 128),
                                              batch_size=1, subset="validation")

# for debugging
# image, label = train_generator.next()
# for i in range(1):
#
#     print(label[i])
#     print(image[i])
#     print(image[i].shape)
#     plt.imshow(image[i])
#     plt.show()

# Compile model
model = create_model()
opt = Adam()
model.compile(loss="mean_squared_error", optimizer=opt, metrics=['accuracy', "mean_squared_error"])

model.summary()

# Train model

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples,
    epochs=200)


# Save Model and results
model.save("vile_counter.h5")
pd.DataFrame(history.history).to_parquet("vile_counter_results.parquet")
