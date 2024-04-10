import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import logging
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from rembg import remove
import requests



app = Flask(__name__)

model = load_model('cinnamease-final.h5')#cinnamease_best_model.h5


uri = "mongodb+srv://chamodadsilva:Chamoda12345@cluster1.kmmlzlq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"

client = MongoClient(uri)

db = client['user_auth']


users_collection = db['users']


@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')

        if users_collection.find_one({'email': email}):
            return jsonify({'error': 'User with this email already exists'})

        hashed_password = generate_password_hash(password)

        user_data = {'name': name, 'email': email, 'password': hashed_password}
        users_collection.insert_one(user_data)

        logging.info('User signed up successfully!')
        return jsonify({'message': 'User signed up successfully'})

    except Exception as e:
        logging.error(f'An error occurred during signup: {str(e)}')
        return jsonify({'error': 'An error occurred during signup'})

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')

        user = users_collection.find_one({'email': email})

        if not user or not check_password_hash(user['password'], password):
            return jsonify({'error': 'Invalid email or password'})

        logging.info('User logged in successfully!')
        return jsonify({'message': 'User logged in successfully'})

    except Exception as e:
        logging.error(f'An error occurred during login: {str(e)}')
        return jsonify({'error': 'An error occurred during login'})




import requests

def process_image(image_data, background_color="black"):

    api_endpoint = "https://api.remove.bg/v1.0/removebg"

    api_key = "EqQqyYgWBiBh98EXqw35pcNr"

    response = requests.post(api_endpoint,
                             files={'image_file': image_data},
                             data={'size': 'auto'},
                             headers={'X-API-Key': api_key})

    if response.status_code == 200:
        output = io.BytesIO(response.content)

        foreground_img = Image.open(output)

        target_size = (256, 256)
        foreground_img = foreground_img.resize(target_size)

        composite_image = Image.new("RGB", foreground_img.size, color=background_color)

        composite_image.paste(foreground_img, (0, 0), mask=foreground_img)

        img_array = image.img_to_array(composite_image)
        img_array = np.expand_dims(img_array, axis=0)

        img_array /= 255.0

        return img_array
    else:
        logging.error(f'Failed to remove background. Status code: {response.status_code}. Response content: {response.content}')
        return None


@app.route("/")
def index():
    return {"message":"hello"}

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'success', 'message': 'Ping successful!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        image_data = file.read()

        img_array = process_image(image_data)

        if img_array is None:
            return jsonify({'error': 'Failed to preprocess image'})

        probability_unmatured = model.predict(img_array)[0][0]

        probability_matured = 1 - probability_unmatured

        margin = 0.2
        if probability_unmatured >= probability_matured + margin:
            maturity_status = 'Unmatured'
        elif probability_matured >= probability_unmatured + margin:
            maturity_status = 'Matured'
        else:
            maturity_status = 'Not Sure'


        result = {
            'probability_matured': float(probability_matured),
            'probability_unmatured': float(probability_unmatured),
            'maturity_status': maturity_status
        }

        logging.info('Prediction successful!')
        return jsonify(result)

    except Exception as e:
        logging.error(f'An error occurred: {str(e)}')
        return jsonify({'error': str(e)})