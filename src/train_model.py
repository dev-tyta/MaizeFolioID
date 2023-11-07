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


#Splitting general dataset into train and test set
splitfolders.ratio(input="../data/raw/data",
                   output= "../data/processed/data",
                   seed= 42, ratio=(.8, .2, 0.0)
                   )
