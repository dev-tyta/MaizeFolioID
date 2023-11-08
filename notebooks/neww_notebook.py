from google.colab import drive
drive.mount('/content/drive')
import tensorflow as tf
import numpy as np
# import splitfolders
# import python_splitter
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import losses
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import RandomOverSampler
from keras.utils import to_categorical

#  splitfolders.ratio("/content/drive/MyDrive/FDD Project/dataset/data",
#                    output="/content/drive/MyDrive/FDD Project/dataset/out",
#                    seed=42,
#                    ratio=(.8, .2, .0)
#                    )

train_dir = "/content/drive/MyDrive/FDD Project/dataset/corn/train"
test_dir = "/content/drive/MyDrive/FDD Project/dataset/corn/val"

import os
import shutil
from random import choice


data_dir = train_dir
class_names = os.listdir(data_dir)
class_counts = {}

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        num_images = len(os.listdir(class_dir))
        class_counts[class_name] = num_images

print(class_counts)

"""### Oversampling"""

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        num_images = class_counts[class_name]
        images_to_copy = 1100 - num_images
        image_files = os.listdir(class_dir)
        while images_to_copy > 0:
            # Choose a random image to copy
            image_to_copy = choice(image_files)
            # Create a new filename
            new_filename = f"{image_to_copy.split('.')[0]}_copy{images_to_copy}.{image_to_copy.split('.')[1]}"
            # Copy the image
            shutil.copy(os.path.join(class_dir, image_to_copy), os.path.join(class_dir, new_filename))
            images_to_copy -= 1

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        print(f"Class {class_name} has {len(os.listdir(class_dir))} images after oversampling.")

"""### Data Augmentation"""

datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split= 0.2)

datagen_validation = ImageDataGenerator(rescale=1./255)

train_generator= datagen_train.flow_from_directory(
    train_dir,
    target_size=(299,299),
    class_mode = "categorical",
    subset="training",
)

val_generator = datagen_train.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    class_mode= "categorical",
    subset="validation"
)

test_generator = datagen_validation.flow_from_directory(
    test_dir,
    target_size = (224,224),
)

"""## Model Training

### VGG16
"""

base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = True
set_trainable = False
for layer in base_model.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Create the sequential model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.summary()

model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

history = model.fit(train_generator,
                    batch_size=54,
                    epochs=50,
                    validation_data=val_generator
                    )

"""### InceptionV3 Model"""

base2 = InceptionV3(weights="imagenet",include_top=False ,input_shape=(299,299,3))
# Set the bottom 10 layers to be trainable
for layer in base2.layers[-10:]:
    layer.trainable = True

model_inception = Sequential([
    base2,
    GlobalAveragePooling2D(),
    Flatten(),
    BatchNormalization(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    BatchNormalization(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    BatchNormalization(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    BatchNormalization(),
    Dense(32, activation="relu"),
    Dropout(0.5),
    Dense(4, activation="softmax")
])

model_inception.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
                )

model_inception.summary()

history_in = model_inception.fit(train_generator,
                     epochs = 50,
                     validation_data = val_generator
                     )

"""### Model Saving"""

model.save("model_vgg.h5", save_format="h5")

model_inception("model_inception.h5", save_format="h5")
