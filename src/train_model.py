from .model_architecture import build_cnn_model
from .utils import create_generators
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from config import EPOCHS, MODEL_DIR, CHECKPOINT_DIR
import os

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def main():
    train_gen, val_gen = create_generators()
    model = build_cnn_model()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ModelCheckpoint(filepath=str(CHECKPOINT_DIR / 'best_model.h5'), monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(str(MODEL_DIR / 'cnn_crumb_rubber.h5'))


if __name__ == '__main__':
    main()
