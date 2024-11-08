import cv2
import numpy as np
from sklearn.decomposition import PCA

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Initialize a list to collect face images
face_images = []

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region and resize it to a fixed size (e.g., 100x100)
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (100, 100))

        # Flatten the face image and add it to the list
        face_images.append(face.flatten())

    # Write the frame to the output file
    out.write(frame)

    # Display the captured frame with rectangles around faces
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()

# Step 4: Apply PCA on Collected Face Images
if face_images:
    # Convert the list of face images to a NumPy array
    face_images = np.array(face_images)

    # Apply PCA to reduce the dimensionality of the face images
    pca = PCA(n_components=50)  # Adjust the number of components as needed
    principal_components = pca.fit_transform(face_images)

    # Print the explained variance ratio to see how much information is retained
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)

    # You can use the principal components for face recognition or further analysis
else:
    print("No faces were detected for PCA.")
