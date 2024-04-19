import cv2
import numpy as np
import pyautogui

# Constants for gesture detection
MIN_CONTOUR_AREA = 1000  # Minimum contour area to be considered as a hand
GESTURE_AREA = 10000  # Area threshold for gesture detection
GESTURE_TIMEOUT = 10  # Timeout in frames for recognizing gestures
GESTURE_RECOGNIZED = False  # Flag to indicate if a gesture has been recognized
GESTURE_FRAME_COUNT = 0  # Counter for frames since last recognized gesture

# Function to perform finger detection and gesture recognition
def detect_fingers(frame):
    global GESTURE_RECOGNIZED, GESTURE_FRAME_COUNT

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Threshold the image to obtain binary image
    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours to find the hand
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_CONTOUR_AREA:
            # Check if the hand is covering a large area
            if area > GESTURE_AREA:
                GESTURE_RECOGNIZED = True

# Function to process recognized gesture
def process_gesture():
    global GESTURE_RECOGNIZED, GESTURE_FRAME_COUNT

    if GESTURE_RECOGNIZED:
        GESTURE_FRAME_COUNT += 1
        if GESTURE_FRAME_COUNT >= GESTURE_TIMEOUT:
            pyautogui.hotkey('space')  # Pause or resume video
    else:
        GESTURE_FRAME_COUNT = 0

    # Reset gesture recognition
    GESTURE_RECOGNIZED = False

# Main function to capture video feed and perform actions
def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect fingers and perform gesture recognition
        detect_fingers(frame)

        # Process recognized gesture
        process_gesture()

        # Display the frame
        cv2.imshow('Finger Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
