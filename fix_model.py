from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import os

# Load the existing model
model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found.")
else:
    model = load_model(model_path)

    # Define a new input layer with the correct shape (150, 150, 3)
    new_input = Input(shape=(150, 150, 3))
    new_model = Model(inputs=new_input, outputs=model(new_input))

    # Save the updated model
    save_path = os.path.join(os.path.dirname(__file__), 'new_model.h5')
    new_model.save(save_path)
    print(f"✅ Model re-saved successfully as {save_path}")
