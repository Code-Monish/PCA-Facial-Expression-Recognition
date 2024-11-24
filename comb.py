# import cv2
# import mediapipe as mp

# # Initialize MediaPipe Face Detection and Face Mesh
# mp_face_detection = mp.solutions.face_detection
# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils

# # Define indices for specific landmarks to draw
# landmark_indices_to_draw = [
#     33,  # Nose tip
#     61,  # Left eye inner corner
#     160, # Right eye inner corner
#     63,  # Left eye outer corner
#     144, # Right eye outer corner
#     13,  # Left eyebrow inner
#     14,  # Left eyebrow outer
#     43,  # Right eyebrow inner
#     44,  # Right eyebrow outer
#     78,  # Mouth left corner
#     308, # Mouth right corner
# ]

# # Start video capture
# camera = cv2.VideoCapture(0)

# if not camera.isOpened():
#     print("Error: Could not open the camera.")
#     exit()

# # Initialize face detection and face mesh
# with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection, \
#      mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

#     while True:
#         ret, frame = camera.read()
#         if not ret:
#             print("Error: Failed to grab the frame.")
#             break

#         # Convert the frame to RGB for MediaPipe processing
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         rgb_frame.flags.writeable = False

#         # Detect faces
#         results = face_detection.process(rgb_frame)

#         # Extract and print face region matrix
#         if results.detections:
#             for detection in results.detections:
#                 bboxC = detection.location_data.relative_bounding_box
#                 h, w, _ = frame.shape
#                 x_min = int(bboxC.xmin * w)
#                 y_min = int(bboxC.ymin * h)
#                 width = int(bboxC.width * w)
#                 height = int(bboxC.height * h)

#                 # Ensure bounding box dimensions are within the frame
#                 x_min = max(0, x_min)
#                 y_min = max(0, y_min)
#                 x_max = min(w, x_min + width)
#                 y_max = min(h, y_min + height)

#                 # Extract face region matrix (ROI)
#                 face_matrix = frame[y_min:y_max, x_min:x_max]

#                 # Print the matrix of the detected face region
#                 print("Face Region Matrix:")
#                 print(face_matrix)

#                 # Draw bounding box around the face
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

#         # Process face mesh and draw selected landmarks
#         results_mesh = face_mesh.process(rgb_frame)

#         if results_mesh.multi_face_landmarks:
#             for face_landmarks in results_mesh.multi_face_landmarks:
#                 for index in landmark_indices_to_draw:
#                     landmark = face_landmarks.landmark[index]
#                     h, w, _ = frame.shape
#                     x = int(landmark.x * w)
#                     y = int(landmark.y * h)
#                     cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Draw selected landmarks

#         # Display the frame
#         cv2.imshow('Real-Time Face Mesh Detection', frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# def extract_important_features(face_region):
#     # Step 1: Convert the image to grayscale
#     gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

#     # Step 2: Apply edge detection to highlight key features
#     edges = cv2.Canny(gray_face, 100, 200)

#     # Step 3: Blend the grayscale image with the edges to emphasize key features
#     # Use a weighted combination of the original grayscale and the edges
#     # Adjust the weights to balance between the original details and the edges
#     blended_image = cv2.addWeighted(gray_face, 0.7, edges, 1, 0)

#     return blended_image

# # Use the modified function to get the reduced face matrix
# reduced_face_matrix = extract_important_features(face_matrix)
# print("reduced image\n", reduced_face_matrix)

# # Display the blended image with emphasized features
# cv2.imshow("Reduced Face Matrix with Features", reduced_face_matrix)
# cv2.waitKey(0)

# # Release the camera and close windows
# camera.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np

class VideoFeatureExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_features_from_camera(self):
        cap = cv2.VideoCapture(0)  # Use camera index 0
        features_list = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with Mediapipe
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                features = self._calculate_features(landmarks)
                if features is not None:
                    features_list.append(features)

            # Display the frame (Optional: Comment out if not needed)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return np.array(features_list)

    def _calculate_features(self, landmarks):
        try:
            # Key facial landmarks (with safe indexing)
            landmarks_indices = {
                'right_eyebrow': 105,
                'left_eyebrow': 334,
                'right_eye_right': 33,
                'right_eye_left': 133,
                'right_eye_top': 159,
                'right_eye_bottom': 145,
                'left_eye_right': 362,
                'left_eye_left': 263,
                'left_eye_top': 386,
                'left_eye_bottom': 374,
                'mouth_right': 78,
                'mouth_left': 308,
                'mouth_top': 13,
                'mouth_bottom': 14
            }

            # Extract landmark coordinates
            def get_landmark(name):
                idx = landmarks_indices.get(name)
                return np.array([landmarks[idx].x, landmarks[idx].y]) if idx is not None and idx < len(landmarks) else None

            # Calculate facial features
            def calculate_distance(point1, point2):
                return np.linalg.norm(point1 - point2) if point1 is not None and point2 is not None else 0

            # Compute specific facial measurements
            features = [
                calculate_distance(get_landmark('right_eye_bottom'), get_landmark('right_eye_top')),  # Right eye height
                calculate_distance(get_landmark('left_eye_bottom'), get_landmark('left_eye_top')),    # Left eye height
                calculate_distance(get_landmark('mouth_right'), get_landmark('mouth_left')),          # Mouth width
                calculate_distance(get_landmark('mouth_top'), get_landmark('mouth_bottom')),          # Mouth height
                calculate_distance(get_landmark('right_eyebrow'), get_landmark('right_eye_top')),    # Right eyebrow to eye distance
                calculate_distance(get_landmark('left_eyebrow'), get_landmark('left_eye_top'))        # Left eyebrow to eye distance
            ]

            return features

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None