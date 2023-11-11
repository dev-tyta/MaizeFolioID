import os
import shutil
from random import choice
import tensorflow as tf
import numpy as np
# import splitfolders
# import python_splitter
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import losses
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

splitfolders.ratio("/content/drive/MyDrive/FDD Project/dataset/data",
                   output="/content/drive/MyDrive/FDD Project/dataset/out",
                   seed=42,
                   ratio=(.6, .2, .2)
                   )

train_dir = "../data/processed/corn/train"
test_dir = "../data/processed/corn/test"
val_dir = "../data/processed/corn/val"

data_dir = train_dir
class_names = os.listdir(train_dir)
class_counts = {}

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        num_images = len(os.listdir(class_dir))
        class_counts[class_name] = num_images

print(class_counts)

"""### Data Augmentation"""

datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

datagen_validation = ImageDataGenerator(rescale=1./255)

train_generator = datagen_train.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    class_mode="categorical",
    subset="training",
)

val_generator = datagen_train.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    class_mode="categorical"
)

test_generator = datagen_validation.flow_from_directory(
    test_dir,
    target_size=(224, 224)
)

"""## Model Training

### VGG16
"""

base_model = VGG16(include_top=False, weights='imagenet', input_shape=(260, 260, 3))
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

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

history = model.fit(train_generator,
                    epochs=30,
                    validation_data=val_generator
                    )

"""### InceptionV3 Model"""
"""
base2 = InceptionV3(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
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
                                 epochs=50,
                                 validation_data=val_generator
                                 )
"""
"""### Model Saving"""

model.save("model_vgg.h5", save_format="h5")

# model_inception("model_inception.h5", save_format="h5")
