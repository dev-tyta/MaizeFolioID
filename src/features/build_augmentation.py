from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

def augmentation(train_dir, test_dir):
    train_generator = gen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        class_mode = "categorical",
        subset="training",
        batch_size =1,
        seed=2020
    )

    val_generator = gen.flow_from_directory(
        train_dir,
        target_size = (224, 224),
        class_mode = "categorical",
        subset="validation"
    )

    test_generator = test_gen.flow_from_directory(
        test_dir,
        target_size= (224, 224)
    )

    return train_generator, val_generator, test_generator
