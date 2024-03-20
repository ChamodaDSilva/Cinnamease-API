import logging
import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from rembg import remove
from PIL import Image
import io

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load the pre-trained H5 model
model = load_model('cinnamease_best_model.h5')

# Function to remove background from an image and add black background
def process_image(image_data, background_color="black"):
    # Remove background
    output = remove(image_data)

    # Convert bytes to image
    foreground_img = Image.open(io.BytesIO(output))

    # Resize the image to 256x256 pixels
    target_size = (256, 256)
    foreground_img = foreground_img.resize(target_size)

    # Create a new image with a black background of the same size as the foreground image
    composite_image = Image.new("RGB", foreground_img.size, color=background_color)

    # Paste the foreground image with the removed background onto the composite image
    composite_image.paste(foreground_img, (0, 0), mask=foreground_img)

    # Convert the composite image to numpy array
    img_array = image.img_to_array(composite_image)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image
    img_array /= 255.0

    return img_array


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'success', 'message': 'Ping successful!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the 'file' key is in the request files
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Read image file as bytes
        image_data = file.read()

        # Preprocess the image
        img_array = process_image(image_data)

        # Make predictions using the loaded model
        prediction = model.predict(img_array)

        # Assuming prediction is a single value between 0 and 1
        maturity_score = prediction[0][0]

        # Defined a threshold for maturity
        maturity_threshold = 0.5

        # Determine maturity status
        maturity_status = 'Unmatured' if maturity_score > maturity_threshold else 'Matured'

        result = {
            'maturity_score': float(maturity_score),
            'maturity_status': maturity_status
        }

        logging.info('Prediction successful!')
        return jsonify(result)

    except Exception as e:
        logging.error(f'An error occurred: {str(e)}')
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    logging.info('Server Started!')
    app.run(debug=True)
