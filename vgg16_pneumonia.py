from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense

# Load the pre-trained VGG16 model (excluding the top fully connected layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Flatten the output layer
x = Flatten()(base_model.output)

# Add a fully connected layer
x = Dense(256, activation='relu')(x)

# Add an output layer for binary classification (pneumonia detection)
x = Dense(1, activation='sigmoid')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=x)

# Print summary
print("VGG16 Model Loaded Successfully!")
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save the model
model.save("vgg16_pneumonia.h5")
print("Model saved as vgg16_pneumonia.h5")
