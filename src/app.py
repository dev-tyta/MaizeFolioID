import streamlit as st
import numpy as np
from io import StringIO
import time
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import io


def preprocess(img_content, target_shape):
    img = Image.open(io.BytesIO(img_content)).resize(target_shape)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


MODEL_PATH = "../models/model_1.h5"
model = load_model(MODEL_PATH)


def prediction(model, x):
    pred = model.predict(x)
    return pred


def classifier(pred):
    class_name = ["Blight", "Common Rust", "Gray Leaf Spot", "Healthy"]
    predicted_class = class_name[np.argmax(pred)]

    return predicted_class


def run():
    st.title("MaizeFolioID")
    st.write("Introducing 'MaizeFolioID': This application applies an advanced "
             "image recognition technology to accurately classify and identify various foliar "
             "diseases affecting maize leaves. " 
             "Simply capture a photo of the leaf, and let MaizeFolioID analyze and detect type in seconds, "
             "helping farmers make informed decisions for healthier crop management.")
    img_in = st.file_uploader(label="Upload the photo of your maize leaf here.", type=["png", "jpg", "jpeg"])
    if img_in is not None:
        img_content = img_in.read()
        # img_path = StringIO(img_content)
        processed_img = preprocess(img_content, target_shape=(224, 224))
        img_pred = prediction(model, x=processed_img)
        class_name = classifier(img_pred)
    
        with st.spinner(text="Detecting Diseases..."):
            time.sleep(10)
        st.write(f"The uploaded maize leaf belongs to the {class_name} class.")


if __name__ == "__main__":
    run()
