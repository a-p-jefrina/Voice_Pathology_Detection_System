# backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
from tensorflow.keras.models import load_model
import librosa

app = Flask(__name__)
CORS(app)

model = load_model("dysarthria_model.h5")

CLASS_LABELS = {0: "Normal", 1: "Vox-Senilis", 2: "Laryngozele"}

def extract_mfcc_2d(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfccs.T

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    mfcc_features = extract_mfcc_2d(filepath)
    target_length = 7589
    current_length = mfcc_features.shape[0]

    if current_length < target_length:
        mfcc_features = np.pad(mfcc_features, ((0, target_length - current_length), (0, 0)), mode='constant')
    else:
        mfcc_features = mfcc_features[:target_length, :]

    mfcc_features = np.expand_dims(mfcc_features, axis=-1)
    test_input = np.expand_dims(mfcc_features, axis=0)
    prediction = model.predict(test_input)
    predicted_class = np.argmax(prediction)
    result = CLASS_LABELS[predicted_class]

    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
