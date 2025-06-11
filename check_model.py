from tensorflow.keras.models import load_model

model_path = r"C:\Users\91992\OneDrive\Desktop\pneumonia-detection-master\model.h5"

model = load_model(model_path)

print("Model Summary:")
model.summary()
