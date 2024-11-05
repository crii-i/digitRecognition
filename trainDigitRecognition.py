import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization

# Set up the path to your dataset
# data_path = 'augmentedTrainingSet/'  # Replace with your directory path if needed
data_path = os.path.join('recognitionSoftware', 'augmentedTrainingSet3')

"""
.########.....###....########....###.....######...########.##....##.########.########.....###....########.####..#######..##....##
.##.....##...##.##......##......##.##...##....##..##.......###...##.##.......##.....##...##.##......##.....##..##.....##.###...##
.##.....##..##...##.....##.....##...##..##........##.......####..##.##.......##.....##..##...##.....##.....##..##.....##.####..##
.##.....##.##.....##....##....##.....##.##...####.######...##.##.##.######...########..##.....##....##.....##..##.....##.##.##.##
.##.....##.#########....##....#########.##....##..##.......##..####.##.......##...##...#########....##.....##..##.....##.##..####
.##.....##.##.....##....##....##.....##.##....##..##.......##...###.##.......##....##..##.....##....##.....##..##.....##.##...###
.########..##.....##....##....##.....##..######...########.##....##.########.##.....##.##.....##....##....####..#######..##....##
"""

# Image dimensions
# image_size = (64, 64)  # Resizing to 64x64 
# batch_size = 32        # Adjust batch size if needed
image_size = (128, 128)  # Resizing to 64x64 
batch_size = 32        # Adjust batch size if needed

# Set up the ImageDataGenerator with augmentations
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # Normalize pixel values between 0 and 1
    rotation_range=1,          # Small rotation for slight variations
    brightness_range=[0.8, 1.2], # Vary brightness to simulate lighting changes
    width_shift_range=0.01,      # Shift images horizontally by up to 10%
    height_shift_range=0.01,     # Shift images vertically by up to 10%
    # zoom_range=0.05,             # Zoom in or out by up to 10%
    # shear_range=0.05,            # Shear by up to 10%
    fill_mode='nearest',        # Fill empty pixels with nearest value
    # contrast_stretching=True,
    validation_split=0.3        # 20% for validation
)

# Set up the training and validation data generators
train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=image_size,
    color_mode='grayscale',     # Grayscale since images have only one channel
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

"""
..######..##....##.##....##
.##....##.###...##.###...##
.##.......####..##.####..##
.##.......##.##.##.##.##.##
.##.......##..####.##..####
.##....##.##...###.##...###
..######..##....##.##....##
"""

# Define the model architecture with an explicit Input layer
# model = Sequential([
#     Input(shape=(64, 64, 1)),  # Define the input shape here
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(10, activation='softmax')  # 10 output units for digits 0-9
# ])

model = Sequential([
    Input(shape=(128, 128, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model is compiled and ready for training.")

# Train the model
epochs = 3  # Set the number of epochs as needed
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Optional: Save the model after training
# model.save('digitRecognitionModel.h5')
model.save('recognitionSoftware/digitRecognitionModel3.keras')
print("Model has been trained and saved.")