import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import (TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE,
                    AUGMENTATION_PARAMS, SEED, MODEL_DIR)
import numpy as np
from PIL import Image


def create_generators():
    train_datagen = ImageDataGenerator(rescale=1./255, **AUGMENTATION_PARAMS)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=SEED
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=SEED
    )

    return train_gen, val_gen


def load_trained_model():
    model_path = MODEL_DIR / 'cnn_crumb_rubber.h5'
    return tf.keras.models.load_model(model_path) if model_path.exists() else None


def preprocess_image(img: Image.Image):
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr
