import streamlit as st
import time
from src.models.get_model import download_model
from src.data.preprocess_img import preprocess
from src.models.predict_model import prediction, classifier


model = download_model("Testys/MaizeFolioID", "model_2.h5")


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
        processed_img = preprocess(img_content, target_shape=(224, 224))
        img_pred = prediction(model, x=processed_img)
        class_name = classifier(img_pred)
    
        with st.spinner(text="Detecting Diseases..."):
            time.sleep(10)
        st.write(f"The uploaded maize leaf belongs to the {class_name} class.")


if __name__ == "__main__":
    run()
