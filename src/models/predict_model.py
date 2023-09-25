import numpy as np


def prediction(mod, x):
    pred = mod.predict(x)
    return pred


def classifier(pred):
    class_name = ["Blight", "Common Rust", "Gray Leaf Spot", "Healthy"]
    predicted_class = class_name[np.argmax(pred)]

    return predicted_class
