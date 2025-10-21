from src.utils import preprocess_image, load_trained_model
from PIL import Image
import argparse
import numpy as np


CLASS_NAMES = ['grade_a', 'grade_b', 'grade_c']


def predict_image(image_path: str):
    model = load_trained_model()
    if model is None:
        raise RuntimeError("Model belum dilatih. Jalankan train_model.py terlebih dahulu.")
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess_image(img)
    preds = model.predict(input_tensor)
    pred_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return CLASS_NAMES[pred_idx], confidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()

    label, conf = predict_image(args.image)
    print({"class": label, "confidence": conf})
