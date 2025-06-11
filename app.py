# -------------------------------
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
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

print("Loading Pre-trained Model ...")
model = load_model(r"C:\Users\91992\OneDrive\Desktop\pneumonia-detection-master\vgg16_pneumonia.h5")
print("Model Loaded Successfully!")

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
