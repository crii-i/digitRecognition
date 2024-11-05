import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model
import imutils  # Make sure imutils is installed for sorting contours

# Path to save/load ROI coordinates
roi_file_path = 'recognitionSoftware/roi_coordinates1.json'

# Load the trained CNN model
model = load_model('recognitionSoftware/singleDigitRecognitionModel2.keras')

# Set the desired width for displaying the image during ROI selection
display_width = 1400

# Function to manually select ROIs and save them
def select_and_save_rois(image_path):
    # Load the original image
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]

    # Calculate the scale factor to resize the image
    scale_factor = display_width / original_width
    display_height = int(original_height * scale_factor)

    # Resize the image for display
    image_resized = cv2.resize(image, (display_width, display_height))

    # List to store coordinates of selected ROIs in the original image scale
    rois = []

    # Manually select 3 ROIs in the resized image
    for i in range(1):
        roi = cv2.selectROI("Select Display Region", image_resized, fromCenter=False, showCrosshair=True)
        
        # Scale the ROI coordinates back up to the original image size
        roi_original = {
            'x': int(roi[0] / scale_factor),
            'y': int(roi[1] / scale_factor),
            'w': int(roi[2] / scale_factor),
            'h': int(roi[3] / scale_factor)
        }
        rois.append(roi_original)
        print(f"ROI {i+1} selected: {roi_original}")

    cv2.destroyAllWindows()
    
    # Save ROIs to JSON file
    with open(roi_file_path, 'w') as file:
        json.dump(rois, file)
    print(f"Saved ROI coordinates to {roi_file_path}")

# Function to load ROIs from JSON file
def load_rois():
    with open(roi_file_path, 'r') as file:
        rois = json.load(file)
    return rois

"""
.##..............#####..
.##....##.......##...##.
.##....##......##.....##
.##....##......##.....##
.#########.....##.....##
.......##..###..##...##.
.......##..###...#####..
"""

def process_image_with_improved_contours(image_path, rois, output_image_path):
    image = cv2.imread(image_path)
    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for idx, roi in enumerate(rois):
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        
        # Crop each display area using saved coordinates
        display_region = gray[y:y + h, x:x + w]

        # Apply Gaussian blur followed by Otsu's thresholding
        blurred = cv2.GaussianBlur(display_region, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Display intermediate thresholded result
        cv2.imshow(f"Thresholded - ROI {idx+1}", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Morphological transformations to improve contour continuity
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Larger kernel
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        # Find contours on the processed image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        digit_contours = []

        # Adjust size constraints to catch more contours
        min_digit_width, max_digit_width = 10, w // 2
        min_digit_height, max_digit_height = 30, h

        for c in cnts:
            (dx, dy, dw, dh) = cv2.boundingRect(c)
            # Filter contours by width and height only
            if min_digit_width <= dw <= max_digit_width and min_digit_height <= dh <= max_digit_height:
                digit_contours.append(c)

        # Sort the contours from left to right to ensure digits are in order
        digit_contours = sorted(digit_contours, key=lambda c: cv2.boundingRect(c)[0])

        digits = []
        for c in digit_contours:
            (dx, dy, dw, dh) = cv2.boundingRect(c)
            digit = thresh[dy:dy + dh, dx:dx + dw]

            # Resize the digit to 64x64 for CNN input
            digit_resized = cv2.resize(digit, (64, 64))
            digit_resized = digit_resized.astype("float32") / 255.0  # Normalize to match CNN input
            digit_resized = np.expand_dims(digit_resized, axis=-1)  # Add channel dimension for CNN
            digit_resized = np.expand_dims(digit_resized, axis=0)   # Add batch dimension for prediction

            # Predict the digit using the CNN
            prediction = model.predict(digit_resized)
            # predicted_digit = np.argmax(prediction, axis=1)[0]
            predicted_digit = np.argmax(prediction)
            digits.append(predicted_digit)

            # Draw bounding box and prediction on the output image
            cv2.rectangle(output_image, (x + dx, y + dy), (x + dx + dw, y + dy + dh), (0, 255, 0), 2)
            cv2.putText(output_image, str(predicted_digit), (x + dx, y + dy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Print the detected number for each display
        print(f"Detected number for display {idx+1}: {''.join(map(str, digits))}")

    # Save the output image with bounding boxes and predictions
    cv2.imwrite(output_image_path, output_image)
    print(f"Saved improved output image with predictions to {output_image_path}")



# Main script to run once
image_path = 'recognitionSoftware/imgs/IMG_2419.jpg'
output_image_path = 'recognitionSoftware/output_with_predictions.jpg'

# Check if ROI coordinates are already saved
try:
    rois = load_rois()
    print("Loaded ROI coordinates from file.")
except (FileNotFoundError, json.JSONDecodeError):
    print("ROI file not found or invalid. Please select ROIs manually.")
    select_and_save_rois(image_path)
    rois = load_rois()

# Process the image with saved ROIs and save the result
process_image_with_improved_contours(image_path, rois, output_image_path)