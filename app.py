from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf 
from tqdm import tqdm 
from glob import glob 
import cv2 
import pandas as pd 

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Image preprocessing 
def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0 
    image = image.astype(np.float32)
    return image 

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


path = './'
train_path = os.path.join(path, 'train/*')
test_path = os.path.join(path, 'test/*')
labels_path = os.path.join(path, 'labels.csv')

# Checking the csv file as a Pandas Dataframe
labels_df = pd.read_csv(labels_path)
breed = labels_df["breed"].unique() # stores list of unique breeds
breed2id = {name: i for i, name in enumerate(breed)}
id2breed = {i: name for i, name in enumerate(breed)}

# So, it's a multiclassification problem 
# We use enumerate() over a dictionary that transcribes 
# each breed it's bree2id 
labels = []
breed2id = {name: i for i , name in enumerate(breed)}

ids = glob(train_path) # used to fetch addresses of all images 
# inside the train folder 
# Preprocessing the training data 
for image_id in ids:
    image_id = image_id.split('\\')[-1].split('.')[0]
    breed_name = list(labels_df[labels_df.id == image_id]['breed'])[0]
    breed_idx = breed2id[breed_name]
    labels.append(breed_idx)

ids = ids[:1000]
labels = labels[:1000]

# Load your trained model
model = tf.keras.models.load_model('model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')




def model_predict(img_path, model):
    image = read_image(img_path, 224)
    image = np.expand_dims(image, axis = 0)
    pred = model.predict(image)[0]
    return pred 


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        label_idx = np.argmax(preds)
        # this string will contain the name of the breed of the dog 
        breed_name = id2breed[label_idx]
        breed_name = breed_name.split('_')
        breed = ' '.join([str(elem) for elem in breed_name]).capitalize()
        return (breed)

    return None


if __name__ == '__main__':
    app.run(debug=True)

