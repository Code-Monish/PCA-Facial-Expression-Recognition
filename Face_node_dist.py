import cv2
import mediapipe as mp
import os
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Function to calculate distances between landmarks and draw them on the image
def calculate_and_draw(image_path, color):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize face mesh
    with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get the coordinates for the selected landmarks
                eye_left = face_landmarks.landmark[159]  # Left eye outer corner
                chin = face_landmarks.landmark[152]  # Chin
                mouth_left = face_landmarks.landmark[78]  # Left mouth corner
                mouth_right = face_landmarks.landmark[308]  # Right mouth corner

                # Convert normalized coordinates to pixel values
                h, w, _ = image.shape
                eye_left_coords = (int(eye_left.x * w), int(eye_left.y * h))
                chin_coords = (int(chin.x * w), int(chin.y * h))
                mouth_left_coords = (int(mouth_left.x * w), int(mouth_left.y * h))
                mouth_right_coords = (int(mouth_right.x * w), int(mouth_right.y * h))

                # Calculate distances
                eye_to_chin_distance = np.linalg.norm(np.array(eye_left_coords) - np.array(chin_coords))
                mouth_width_distance = np.linalg.norm(np.array(mouth_left_coords) - np.array(mouth_right_coords))

                # Draw landmarks and distances on the image
                cv2.circle(image, eye_left_coords, 5, color, -1)  # Eye
                cv2.circle(image, chin_coords, 5, color, -1)  # Chin
                cv2.circle(image, mouth_left_coords, 5, color, -1)  # Left mouth corner
                cv2.circle(image, mouth_right_coords, 5, color, -1)  # Right mouth corner

                # Optionally, draw lines between the points
                cv2.line(image, eye_left_coords, chin_coords, color, 1)
                cv2.line(image, mouth_left_coords, mouth_right_coords, color, 1)

                return (eye_to_chin_distance, mouth_width_distance, image)

    return None

# Function to process images in two folders
def process_images_in_folders(folder1_path, folder2_path):
    distances_list = []

    # Process images in the first folder (yellow)
    for image_name in os.listdir(folder1_path):
        if image_name.endswith('.jpg') or image_name.endswith('.jpeg'):
            image_path = os.path.join(folder1_path, image_name)
            result = calculate_and_draw(image_path, (0, 255, 255))  # Yellow color
            if result is not None:
                distances_list.append(result[:-1])  # Append distances only

    # Process images in the second folder (red)
    for image_name in os.listdir(folder2_path):
        if image_name.endswith('.jpg') or image_name.endswith('.jpeg'):
            image_path = os.path.join(folder2_path, image_name)
            result = calculate_and_draw(image_path, (0, 0, 255))  # Red color
            if result is not None:
                distances_list.append(result[:-1])  # Append distances only

    return distances_list

# Paths to the folders containing images
folder1_path = 'Emotions Dataset/test/happy'  # Change this to your first folder path
folder2_path = 'Emotions Dataset/test/sad'  # Change this to your second folder path

# Call the function to process images and get distances
distances = process_images_in_folders(folder1_path, folder2_path)

# Prepare data for plotting
eye_to_chin_distances = [d[0] for d in distances]
mouth_width_distances = [d[1] for d in distances]

# Create a scatter plot
# Create a scatter plot
plt.scatter(eye_to_chin_distances[:len(distances)//2], mouth_width_distances[:len(distances)//2], color='yellow', label='Folder 1')
plt.scatter(eye_to_chin_distances[len(distances)//2:], mouth_width_distances[len(distances)//2:], color='red', label='Folder 2')

# Set labels and title
plt.xlabel('Eye to Chin Distance')
plt.ylabel('Mouth Width Distance')
plt.title('Eye to Chin Distance vs. Mouth Width Distance')
plt.legend()
plt.axis('equal')
plt.grid()

# Show the plot
plt.show()