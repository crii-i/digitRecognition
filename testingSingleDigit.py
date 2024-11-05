# import os
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# from tkinter import Tk
# from tkinter.filedialog import askopenfilename

# # Load the pre-trained model
# try:
#     model = load_model('recognitionSoftware/digitRecognitionModel.keras')
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)

# # Function to preprocess the image
# def preprocess_image(image_path):
#     try:
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         if image is None:
#             raise ValueError("Image not found or unable to read.")
#         image = cv2.resize(image, (64, 64))  # Resize to 64x64
#         image = image / 255.0  # Normalize pixel values
#         image = np.expand_dims(image, axis=-1)  # Add channel dimension
#         image = np.expand_dims(image, axis=0)  # Add batch dimension
#         return image
#     except Exception as e:
#         print(f"Error preprocessing image: {e}")
#         return None

# # Function to predict the digit
# def predict_digit(image_path):
#     image = preprocess_image(image_path)
#     if image is None:
#         return None
#     try:
#         prediction = model.predict(image)
#         digit = np.argmax(prediction)
#         return digit
#     except Exception as e:
#         print(f"Error predicting digit: {e}")
#         return None

# # Main function
# def main():
#     Tk().withdraw()  # Hide the root window
#     image_path = askopenfilename(title="Select an image of a digit", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
#     if image_path:
#         digit = predict_digit(image_path)
#         if digit is not None:
#             print(f"The predicted digit is: {digit}")
#         else:
#             print("Prediction failed.")
#     else:
#         print("No image selected.")

# if __name__ == "__main__":
#     main()





"""
................................................................................................................................
................................................................................................................................
................................................................................................................................
.#######.#######.#######.#######.#######.#######.#######.#######.#######.#######.#######.#######.#######.#######.#######.#######
................................................................................................................................
................................................................................................................................
................................................................................................................................
"""





import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

# Load the pre-trained model
try:
    model = load_model('recognitionSoftware/digitRecognitionModel.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Function to preprocess the image
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image not found or unable to read.")
        image = cv2.resize(image, (64, 64))  # Resize to 64x64
        image = image / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Function to predict the digit
def predict_digit(image_path):
    image = preprocess_image(image_path)
    if image is None:
        return None
    try:
        prediction = model.predict(image)
        digit = np.argmax(prediction)
        return digit
    except Exception as e:
        print(f"Error predicting digit: {e}")
        return None

# Main function
def main():
    Tk().withdraw()  # Hide the root window
    image_paths = askopenfilenames(title="Select images of digits", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if image_paths:
        predictions = []
        for image_path in image_paths:
            digit = predict_digit(image_path)
            if digit is not None:
                predictions.append(digit)
            else:
                predictions.append("Prediction failed")
        print(f"The predicted digits are: {predictions}")
    else:
        print("No images selected.")

if __name__ == "__main__":
    main()