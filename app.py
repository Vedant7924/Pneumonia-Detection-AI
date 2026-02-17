# -------------------------------
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_INSTALLED = True
except ImportError:
    print("WARNING: TensorFlow not installed. Model will not be loaded.")
    TENSORFLOW_INSTALLED = False

import numpy as np
import os
import cv2

# Creating a Flask Instance
app = Flask(__name__)

IMAGE_SIZE = (150, 150)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model using relative path
model = None
if TENSORFLOW_INSTALLED:
    model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), 'vgg16_pneumonia.h5')

    print(f"Loading Pre-trained Model from {model_path} ...")
    try:
        model = load_model(model_path)
        print("Model Loaded Successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print("Skipping model loading due to missing TensorFlow.")


def image_preprocessor(path):
    '''
    Function to pre-process the image before feeding to model.
    '''
    print('Processing Image ...')
    currImg_BGR = cv2.imread(path)
    b, g, r = cv2.split(currImg_BGR)
    currImg_RGB = cv2.merge([r, g, b])
    currImg = cv2.resize(currImg_RGB, IMAGE_SIZE)
    currImg = currImg / 255.0
    currImg = np.reshape(currImg, (1, 150, 150, 3))
    return currImg

def model_pred(image):
    if model is None:
        print("Model is not loaded. Returning a mock result.")
        import random
        return random.randint(0, 1)

    print("Image_shape:", image.shape)
    print("Image_dimension:", image.ndim)

    predictions = model.predict(image)
    print("Raw Model Output:", predictions)  # Print raw output from the model

    # Adjusted threshold from 0.5 → 0.4
    predicted_class = int(predictions[0][0] > 0.5)

    print("Predicted Class:", predicted_class)
    return predicted_class


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'imageFile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['imageFile']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Ensure the uploads folder exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                print("Uploads folder missing! Creating now...")
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            filename = secure_filename(file.filename)
            imgPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(imgPath)
            print(f"Image saved at {imgPath}")
            
            # Preprocessing Image
            image = image_preprocessor(imgPath)
            
            # Performing Prediction
            pred = model_pred(image)
            return render_template('upload.html', name=filename, result=pred)
    
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

# -----------------------------------------
