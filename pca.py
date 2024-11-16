import cv2
import numpy as np

face_region = []
# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Read and display video frames
while True:
    ret, frame = camera.read()
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

def extract_important_features(face_region):
    # Step 1: Convert the image to grayscale
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply edge detection to highlight key features
    edges = cv2.Canny(gray_face, 100, 200)

    # Step 3: Blend the grayscale image with the edges to emphasize key features
    # Use a weighted combination of the original grayscale and the edges
    # Adjust the weights to balance between the original details and the edges
    blended_image = cv2.addWeighted(gray_face, 0.7, edges, 1, 0)

    return blended_image

# Use the modified function to get the reduced face matrix
reduced_face_matrix = extract_important_features(face_region)
print("reduced image\n", reduced_face_matrix)

# Display the blended image with emphasized features
cv2.imshow("Reduced Face Matrix with Features", reduced_face_matrix)
cv2.waitKey(0)

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
