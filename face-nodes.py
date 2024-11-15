import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define the indices of the landmarks you want to draw
# These indices correspond to the face mesh landmarks
# You can adjust these indices based on your needs
landmark_indices_to_draw = [
    33,  # Nose tip
    61,  # Left eye inner corner
    160, # Right eye inner corner
    63,  # Left eye outer corner
    144, # Right eye outer corner
    13,  # Left eyebrow inner
    14,  # Left eyebrow outer
    43,  # Right eyebrow inner
    44,  # Right eyebrow outer
    78,  # Mouth left corner
    308, # Mouth right corner
]

# Function to process images and detect face landmarks
def process_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize face detection and face mesh
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection, \
         mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

        # Loop through all images in the input folder
        for image_name in os.listdir(input_folder):
            # Check if the file is a JPEG image
            if not (image_name.endswith('.jpg') or image_name.endswith('.jpeg')):
                continue

            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            # Convert the image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False

            # Detect faces
            results = face_detection.process(rgb_image)

            # Draw face detections
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    cv2.rectangle(image, bbox, (255, 0, 0), 2)

            # Process face mesh
            results_mesh = face_mesh.process(rgb_image)

            # Draw selected landmarks
            if results_mesh.multi_face_landmarks:
                for face_landmarks in results_mesh.multi_face_landmarks:
                    for index in landmark_indices_to_draw:
                        landmark = face_landmarks.landmark[index]
                        h, w, _ = image.shape
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # Draw a circle for each landmark

            # Save the output image
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, image)

            # Optionally display the image
            cv2.imshow('Face Detection and Mesh', image)
            cv2.waitKey(100)  # Wait for a short period to display the image

    cv2.destroyAllWindows()

# Define input and output folders
input_folder = 'faces94/male/akatsi'  # Change this to your input folder
output_folder = 'generated_dataset'  # Change this to your output folder

# Process the images
process_images(input_folder, output_folder)