from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import tensorflow as tf

from src.utils import load_trained_model, preprocess_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_trained_model()
class_names = ['grade_a', 'grade_b', 'grade_c']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = Image.open(filepath).convert('RGB')
    input_tensor = preprocess_image(img)
    preds = model.predict(input_tensor)
    pred_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))

    return jsonify({'class': class_names[pred_idx], 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
