from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

# Load the existing model
model_path = r"C:\Users\91992\OneDrive\Desktop\pneumonia-detection-master\model.h5"
model = load_model(model_path)

# Define a new input layer with the correct shape (150, 150, 3)
new_input = Input(shape=(150, 150, 3))
new_model = Model(inputs=new_input, outputs=model(new_input))

# Save the updated model
new_model.save(r"C:\Users\91992\OneDrive\Desktop\pneumonia-detection-master\new_model.h5")
print("✅ Model re-saved successfully as new_model.h5")
