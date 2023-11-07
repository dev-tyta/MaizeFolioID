import tensorflow as tf
import numpy as np
import splitfolders
import python_splitter
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras import losses
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from huggingface_hub import hf_hub_download


# Splitting general dataset into train and test set
splitfolders.ratio(input="../data/raw/data",
                   output= "../data/processed/corn",
                   seed= 42, ratio=(.8, .2, 0.0)
                   )

# Paths to train and test data for preprocessing
train_data = "../data/processed/corn/train"
test_data = "../data/processed/corn/test"

# data preprocessing
gen = ImageDataGenerator(rescale=1./255,
                         rotation_range=40,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         brightness_range=[0.8, 1.2],
                         fill_mode='nearest',
                         validation_split= 0.2
)

test_gen = ImageDataGenerator(
    rescale=1./255
)

train_generator = gen.flow_from_directory(
    train_data,
    target_size=(224,224),
    class_mode = "categorical",
    subset="training",
    batch_size =1,
    seed=2020
)

val_generator = gen.flow_from_directory(
    train_data,
    target_size = (224, 224),
    class_mode = "categorical",
    subset="validation"
)

test_generator = test_gen.flow_from_directory(
    test_data,
    target_size= (224, 224)
)


# Model Training
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
