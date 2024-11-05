import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model

# Paths for model and ROI coordinates
roi_file_path = 'recognitionSoftware/roi_coordinates.json'
model = load_model('recognitionSoftware/singleDigitRecognitionModel.keras')
display_width = 1400

# Step 1: Select four corners of each display with visual feedback
def select_and_save_display_corners(image_path):
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    scale_factor = display_width / original_width
    display_height = int(original_height * scale_factor)
    image_resized = cv2.resize(image, (display_width, display_height))

    displays_corners = []

    for i in range(3):  # Assuming 3 displays
        print(f"Select 4 corners for display {i + 1}")
        selected_points = []

        def on_mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_points.append((x, y))
                cv2.circle(image_resized, (x, y), 5, (0, 255, 0), -1)

                if len(selected_points) > 1:
                    cv2.line(image_resized, selected_points[-2], selected_points[-1], (0, 255, 0), 2)
                
                if len(selected_points) == 4:
                    cv2.line(image_resized, selected_points[3], selected_points[0], (0, 255, 0), 2)
                    cv2.imshow("Select Corners", image_resized)
                    cv2.waitKey(500)
                    cv2.destroyWindow("Select Corners")

        cv2.imshow("Select Corners", image_resized)
        cv2.setMouseCallback("Select Corners", on_mouse_click)
        cv2.waitKey(0)

        if len(selected_points) == 4:
            corners = [
                (int(point[0] / scale_factor), int(point[1] / scale_factor))
                for point in selected_points
            ]
            displays_corners.append(corners)
            print(f"Selected corners for display {i + 1}: {corners}")
        else:
            print("Please select exactly 4 points!")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    with open(roi_file_path, 'w') as file:
        json.dump(displays_corners, file)
    print(f"Saved display corners to {roi_file_path}")

# Step 2: Load the selected display corners from JSON file
def load_display_corners():
    with open(roi_file_path, 'r') as file:
        displays_corners = json.load(file)
    return displays_corners

# Step 3: Apply perspective transformation based on corners
def apply_perspective_transform(image, corners):
    src_pts = np.array(corners, dtype="float32")
    width, height = 200, 100
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped

# Step 4: Process image and show thresholded ROI for debugging
def process_image_with_perspective(image_path, output_image_path):
    image = cv2.imread(image_path)
    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    displays_corners = load_display_corners()

    for idx, corners in enumerate(displays_corners):
        transformed_display = apply_perspective_transform(gray, corners)
        
        blurred = cv2.GaussianBlur(transformed_display, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Display the thresholded image for each ROI
        cv2.imshow(f"Thresholded ROI {idx+1}", thresh)
        cv2.waitKey(500)  # Display for 500ms; adjust as needed

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_contours = []

        min_digit_width, max_digit_width = 10, 50
        min_digit_height, max_digit_height = 30, transformed_display.shape[0]
        min_aspect_ratio, max_aspect_ratio = 0.2, 1.0

        for c in cnts:
            (dx, dy, dw, dh) = cv2.boundingRect(c)
            aspect_ratio = dw / float(dh)
            if (min_digit_width <= dw <= max_digit_width and 
                min_digit_height <= dh <= max_digit_height and
                min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                digit_contours.append(c)

        digit_contours = sorted(digit_contours, key=lambda c: cv2.boundingRect(c)[0])

        digits = []
        for c in digit_contours:
            (dx, dy, dw, dh) = cv2.boundingRect(c)
            digit = thresh[dy:dy + dh, dx:dx + dw]

            digit_resized = cv2.resize(digit, (64, 64))
            digit_resized = digit_resized.astype("float32") / 255.0
            digit_resized = np.expand_dims(digit_resized, axis=-1)
            digit_resized = np.expand_dims(digit_resized, axis=0)

            prediction = model.predict(digit_resized)
            predicted_digit = np.argmax(prediction, axis=1)[0]
            digits.append(predicted_digit)

            cv2.rectangle(output_image, (corners[0][0] + dx, corners[0][1] + dy),
                            (corners[0][0] + dx + dw, corners[0][1] + dy + dh), (0, 255, 0), 2)
            cv2.putText(output_image, str(predicted_digit), (corners[0][0] + dx, corners[0][1] + dy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        print(f"Detected number for display {idx + 1}: {''.join(map(str, digits))}")

    cv2.imwrite(output_image_path, output_image)
    print(f"Saved output image with predictions to {output_image_path}")

    cv2.destroyAllWindows()

# Main part: Run the process
image_path = 'recognitionSoftware/imgs/IMG_2419.jpg'
output_image_path = 'recognitionSoftware/output_with_predictions.jpg'

try:
    displays_corners = load_display_corners()
    print("Loaded display corners from file.")
except (FileNotFoundError, json.JSONDecodeError):
    print("Display corners file not found or invalid. Please select corners manually.")
    select_and_save_display_corners(image_path)
    displays_corners = load_display_corners()

# Process the image and save the result
process_image_with_perspective(image_path, output_image_path)