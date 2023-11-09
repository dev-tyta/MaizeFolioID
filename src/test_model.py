# importing libraries
import numpy as np
# from tensorflow.keras import models
# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from .models.get_model import download_model

model_ = download_model("Testys/MaizeFolioID", "model_2.h5")
img_ = '../data/external/cr_leaf.jpeg'


# preprocessing external test image
def model_test(img_path, model):
    img_path = img_path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    class_names = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']
    prediction = model.predict(x)
    predicted_class_dict = np.argmax(prediction)
    predicted_class = class_names[predicted_class_dict]

    return predicted_class


results = model_test(img_path=img_, model=model_)
print(results)
