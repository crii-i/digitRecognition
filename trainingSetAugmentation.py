import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Path to the original dataset
input_base_path = 'recognitionSoftware/trainingSet'  # Base path for the dataset
output_base_path = 'recognitionSoftware/augmentedTrainingSet3'  # Base path for augmented images

# Ensure the output base directory exists
os.makedirs(output_base_path, exist_ok=True)

# Define your ImageDataGenerator with augmentation options
datagen = ImageDataGenerator(
    rotation_range=0.03,          # Rotate images by up to 20 degrees
    width_shift_range=0.02,      # Shift images horizontally by up to 10% of width
    height_shift_range=0.01,     # Shift images vertically by up to 10% of height
    # zoom_range=,             # Zoom in or out by up to 10%
    # shear_range=0.05,            # Shear by up to 10%
    brightness_range=[0.98, 1.02],# Randomly adjust brightness
    fill_mode='nearest'         # Fill any empty pixels created by transformation
)

# Loop through each subdirectory in the input base directory
for subdir in os.listdir(input_base_path):
    input_path = os.path.join(input_base_path, subdir)
    output_path = os.path.join(output_base_path, subdir)
    
    # Ensure the output subdirectory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Loop through each image in the input subdirectory
    for filename in os.listdir(input_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other formats if needed
            img_path = os.path.join(input_path, filename)
            img = load_img(img_path)  # Load the image
            x = img_to_array(img)  # Convert the image to a numpy array
            x = x.reshape((1,) + x.shape)  # Reshape the array

            # Generate and save 5 augmented images per original image
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_path, save_prefix='aug', save_format='jpeg'):
                i += 1
                if i >= 200:  # Change this number to generate more or fewer images
                    break

print("Augmentation complete.")