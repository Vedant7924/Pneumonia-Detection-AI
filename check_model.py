from tensorflow.keras.models import load_model
import os

model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(__file__), 'vgg16_pneumonia.h5')

print(f"Checking model at: {model_path}")
try:
    model = load_model(model_path)
    print("Model Summary:")
    model.summary()
except Exception as e:
    print(f"Error: {e}")
