import cv2
import mediapipe as mp
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Define the indices of the landmarks for mouth corners and chin
mouth_left_index = 78  # Left mouth corner
mouth_right_index = 308  # Right mouth corner
chin_index = 152  # Chin
eye_left_index = 159  # Left eye outer corner
eye_right_index = 386  # Right eye outer corner

# Function to process images and extract face height and mouth width
def extract_face_data(input_folder):
    face_data = []

    # Initialize face mesh
    with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        for image_name in os.listdir(input_folder):
            if not (image_name.endswith('.jpg') or image_name.endswith('.jpeg')):
                continue

            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            # Convert the image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_mesh = face_mesh.process(rgb_image)

            if results_mesh.multi_face_landmarks:
                for face_landmarks in results_mesh.multi_face_landmarks:
                    # Get the coordinates for the mouth corners, chin, and eyes
                    mouth_left = face_landmarks.landmark[mouth_left_index]
                    mouth_right = face_landmarks.landmark[mouth_right_index]
                    chin = face_landmarks.landmark[chin_index]
                    eye_left = face_landmarks.landmark[eye_left_index]
                    eye_right = face_landmarks.landmark[eye_right_index]

                    # Calculate mouth width and height (eye to chin distance)
                    mouth_width = np.linalg.norm(np.array([mouth_left.x, mouth_left.y]) - np.array([mouth_right.x, mouth_right.y]))
                    eye_to_chin_height = np.linalg.norm(np.array([eye_left.y, eye_left.x]) - np.array([chin.y, chin.x]))

                    # Append the height and mouth width as a tuple
                    face_data.append((eye_to_chin_height, mouth_width))

    return face_data

# Define input folder
input_folder = 'generated_dataset'  # Change this to your output folder from the previous code

# Extract face data
face_data = extract_face_data(input_folder)

# Now you can use the face_data for PCA
if face_data:
    # Convert to numpy array for PCA
    data = np.array(face_data)

    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)

    # Create the scatter plot of the PCA results
    plt.figure(figsize=(10, 8))
    
    # Scatter plot of PCA results
    plt.scatter(pca_result[:, 0], pca_result[:, 1], color='blue', marker='o')

    # Adding titles and labels
    plt.title('PCA Outcomes of Face Heights and Mouth Widths')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Show the plot
    plt.grid()
    plt.show()
else:
    print("No face data extracted.")