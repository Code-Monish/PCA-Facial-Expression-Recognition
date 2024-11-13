import cv2
import numpy as np

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera
camera = cv2.VideoCapture(0)  # '0' is usually the built-in camera

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Read and display video frames
while True:
    ret, frame = camera.read()  # Read a frame
    if not ret:
        print("Error: Failed to grab the frame.")
        break

    # Convert the frame to grayscale (needed for Haar cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces and get face regions as matrices
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Get the face region as a matrix
        face_region = frame[y:y + h, x:x + w]

        # Print the matrix of the detected face
        print("Face Region Matrix:")
        print(face_region)

    # Display the frame with detected faces
    cv2.imshow("Camera", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

def extract_important_features(face_region):
    # Step 1: Convert the image to grayscale
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply edge detection to highlight key features
    edges = cv2.Canny(gray_face, 100, 200)
    
    # Step 3: Retain only the edges in the grayscale image
    # Create a mask to highlight important features
    reduced_matrix = np.where(edges > 0, gray_face, 0)
    
    return reduced_matrix

reduced_face_matrix = extract_important_features(face_region)

# Display the reduced matrix
print("Reduced Face Matrix")
print(reduced_face_matrix)
cv2.waitKey(0)

# Assuming 'reduced_face_matrix' is your 2D matrix from the reduction step

# Normalize the values to range from 0 to 255 if they aren't already
normalized_matrix = cv2.normalize(reduced_face_matrix, None, 0, 255, cv2.NORM_MINMAX)

# Convert the normalized matrix to an 8-bit unsigned integer type
image = normalized_matrix.astype(np.uint8)

# Display the image
cv2.imshow("Extracted Face Image", image)
cv2.waitKey(0)

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
