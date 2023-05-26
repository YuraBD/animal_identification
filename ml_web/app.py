from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import tensorflow as tf
import numpy as np
import os
from get_spectogram import create_spectogram_image, convert_to_mp3
import time

app = Flask(__name__)
animals = ["cat", "dog", "cow", "horse", "crow", "pig"]
model = tf.keras.models.load_model('my_model_last.h5')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return 'No audio file uploaded.', 400

    audio_file = "static/audio.mp3"
    spectorgam_file = "static/spectogram.png"
    audio = request.files['audio']
    filename = secure_filename(audio.filename)
    audio.save(filename)
    convert_to_mp3(filename, audio_file)
    create_spectogram_image(audio_file, spectorgam_file)
    os.remove(filename)

    image = plt.imread(spectorgam_file)
    image = image[:, :, :3]
    zoom_factors = (256 / image.shape[0], 256 / image.shape[1], 1)
    image = zoom(image, zoom_factors, order=1, mode='reflect', cval=0, prefilter=True)
    prediction = model.predict(np.array([image]))
    probability = tf.nn.softmax(prediction)
    predicted_class = np.argmax(probability, axis=1)

    return render_template('prediction.html', prediction=animals[predicted_class[0]], time=time)

if __name__ == '__main__':
    app.run(debug=True)
