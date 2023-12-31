import os
import tensorflow as tf
import numpy as np
import splitfolders
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import losses
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import efficientnet.tfkeras as enet
from collections import Counter
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

splitfolders.ratio("../data/raw/data",
                   output="../data/processed/out",
                   seed=42,
                   ratio=(.6, .2, .2)
                   )

train_dir = "../data/processed/out/train"
test_dir = "../data/processed/out/test"
val_dir = "../data/processed/out/val"

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

datagen_train = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   validation_split=0.0
                                   )

datagen_validation = ImageDataGenerator(rescale=1./255,
                                        validation_split=0.0
                                        )
datagen_test = ImageDataGenerator(rescale=1./255)

train_generator = datagen_train.flow_from_directory(train_dir,
                                                    target_size=(224, 224),
                                                    class_mode="categorical"
                                                    )

val_generator = datagen_validation.flow_from_directory(val_dir,
                                                       target_size=(224, 224),
                                                       class_mode="categorical"
                                                       )

test_generator = datagen_test.flow_from_directory(test_dir,
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
                    epochs=50,
                    validation_data=val_generator
                    )

"""### Model Saving"""

model.save("model_vgg.h5", save_format="h5")

# Removing Inception model from code

# EfficientNet
my_model = enet.EfficientNetB2(input_shape=(260, 260, 3), include_top=False, weights='imagenet')

inputs_1 = tf.keras.Input(shape=(260, 260, 3))
x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(my_model.output)
x = tf.keras.layers.Flatten()(x)
predictors = tf.keras.layers.Dense(4, activation='softmax', name='Predictions')(x)
model_eff = Model(mymodel.input, outputs=predictors)
for layer_eff in model_eff.layers[:1]:
    layer_eff.trainable = False


def scheduler(epoch, lr):
    if epoch < 3:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def callbacks(patience=2):
    early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, min_delta=0.001)
    lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    callbacks_list = [early, lr]
    return callbacks_list


counter = Counter(train_data.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}

model_eff.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"]
                  )

history_eff = model_eff.fit(train_generator,
                            epochs=50,
                            validation_data=val_generator,
                            callbacks=callbacks,
                            class_weight=class_weights
                            )

# Model saving for EfficientNet

model_eff.save("model_eff.h5", save_format="h5")
