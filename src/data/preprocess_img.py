import io
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image


def preprocess(img_content, target_shape):
    img = Image.open(io.BytesIO(img_content)).resize(target_shape)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x
