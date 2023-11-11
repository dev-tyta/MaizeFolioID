from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import losses
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential

"""### InceptionV3 Model"""

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
