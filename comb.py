import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define indices for specific landmarks to draw
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

# Start video capture
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Initialize face detection and face mesh
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to grab the frame.")
            break

        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Detect faces
        results = face_detection.process(rgb_frame)

        # Extract and print face region matrix
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x_min = int(bboxC.xmin * w)
                y_min = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)

                # Ensure bounding box dimensions are within the frame
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_min + width)
                y_max = min(h, y_min + height)

                # Extract face region matrix (ROI)
                face_matrix = frame[y_min:y_max, x_min:x_max]

                # Print the matrix of the detected face region
                print("Face Region Matrix:")
                print(face_matrix)

                # Draw bounding box around the face
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Process face mesh and draw selected landmarks
        results_mesh = face_mesh.process(rgb_frame)

        if results_mesh.multi_face_landmarks:
            for face_landmarks in results_mesh.multi_face_landmarks:
                for index in landmark_indices_to_draw:
                    landmark = face_landmarks.landmark[index]
                    h, w, _ = frame.shape
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Draw selected landmarks

        # Display the frame
        cv2.imshow('Real-Time Face Mesh Detection', frame)

        # Break the loop if 'q' is pressed
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
reduced_face_matrix = extract_important_features(face_matrix)
print("reduced image\n", reduced_face_matrix)

# Display the blended image with emphasized features
cv2.imshow("Reduced Face Matrix with Features", reduced_face_matrix)
cv2.waitKey(0)

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
