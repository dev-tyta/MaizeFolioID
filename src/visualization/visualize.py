import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_accuracy_plot(model):
    # Summarize history for accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.show()


# plotting random images in data
def show_random_image(generator):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 15))

    for i in range(4):
        image = next(generator)[0]

        # Rescale pixel values from [0, 1] to [0, 255]
        image = np.squeeze(image) * 255.0

        # Convert the pixel values to uint8 type
        image = image.astype('uint8')

        # Display the image
        ax[i].imshow(image)
        ax[i].axis('off')


def plot_confusion_matrix(actual, predicted, label=None):
    if label is None:
        label = [3, 2, 1, 0]
    conf = confusion_matrix(actual, predicted, labels=label)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
