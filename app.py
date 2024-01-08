from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Load the trained model
model = load_model('pneumonia_cnn.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        # Preprocess the image
        img = image.load_img(BytesIO(file.read()), target_size=(256, 256), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize

        # Make prediction
        prediction = model.predict(img_array)

        # Decode the prediction
        classes = ['NORMAL', 'PNEUMONIA']
        result = classes[np.argmax(prediction)]

        # Return the result
        return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
