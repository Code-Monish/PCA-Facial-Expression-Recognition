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

        # Ensure face_region is a 3D array (it should already be)
        #if len(face_region.shape) == 3:
            #print("Face Region is a 3D array with shape:", face_region.shape)
        #else:
            #print("Face Region is not a 3D array.")

    # Display the frame with detected faces
    cv2.imshow("Camera", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



def extract_important_features(face_region):
    # Load the Haar cascades for eyes, nose, and mouth
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')

    # Step 1: Convert the image to grayscale
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

    # Step 2: Detect eyes
    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
    if len(eyes) > 0:
        print("Eyes detected.")
    else:
        print("No eyes detected.")

    # Step 3: Detect nose
    noses = nose_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
    if len(noses) > 0:
        print("Nose detected.")
    else:
        print("No nose detected.")

    # Step 4: Detect mouth
    mouths = mouth_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
    if len(mouths) > 0:
        print("Mouth detected.")
    else:
        print("No mouth detected.")

    # Note: The function does not return anything; it only prints detection results.

# The rest of your code remains unchanged.

# Use the modified function to get the reduced face matrix
if face_region.size > 0:  # Check if face_region is not empty
    extract_important_features(face_region)
    #print("Reduced image shape:", reduced_face_matrix.shape)
    #print("Face region shape:", face_region.shape)

    # Display the blended image with emphasized features
    cv2.imshow("Reduced Face Matrix with Features", reduced_face_matrix)
    cv2.waitKey(0)

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()