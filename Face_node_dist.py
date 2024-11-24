import numpy as np
import cv2
import mediapipe as mp
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

class EmotionFeatureExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    def calculate_emotion_features(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        try:
            # Carefully select landmark indices
            # These indices might need adjustment based on your specific Mediapipe version
            a1 = np.array([landmarks[105].x, landmarks[105].y])  # Right eyebrow
            b1 = np.array([landmarks[334].x, landmarks[334].y])  # Left eyebrow

            # Right eye landmarks
            c1 = np.array([landmarks[33].x, landmarks[33].y])    # Right end
            c2 = np.array([landmarks[133].x, landmarks[133].y])  # Left end
            c3 = np.array([landmarks[159].x, landmarks[159].y])  # Top
            c4 = np.array([landmarks[145].x, landmarks[145].y])  # Bottom
            
            # Left eye landmarks
            d1 = np.array([landmarks[362].x, landmarks[362].y])  # Right end
            d2 = np.array([landmarks[263].x, landmarks[263].y])  # Left end
            d3 = np.array([landmarks[386].x, landmarks[386].y])  # Top
            d4 = np.array([landmarks[374].x, landmarks[374].y])  # Bottom

            # Lips landmarks
            f1 = np.array([landmarks[78].x, landmarks[78].y])    # Right end
            f2 = np.array([landmarks[308].x, landmarks[308].y])  # Left end
            f3 = np.array([landmarks[13].x, landmarks[13].y])    # Top
            f4 = np.array([landmarks[14].x, landmarks[14].y])    # Bottom
            f5 = np.array([(f1[0] + f2[0])/2, (f1[1] + f2[1])/2])  # Lip center

            # Calculate dimensions
            eye_height = (np.linalg.norm(c4 - c3) + np.linalg.norm(d4 - d3)) / 2
            eye_width = (np.linalg.norm(c2 - c1) + np.linalg.norm(d2 - d1)) / 2
            mouth_height = np.linalg.norm(f4 - f3)
            mouth_width = np.linalg.norm(f2 - f1)
            
            # Estimate eye center (average of top and bottom points)
            c5 = np.array([(c3[0] + c4[0])/2, (c3[1] + c4[1])/2])
            d5 = np.array([(d3[0] + d4[0])/2, (d3[1] + d4[1])/2])

            eyebrow_to_eye_center = (np.linalg.norm(c5 - a1) + np.linalg.norm(d5 - b1)) / 2
            eye_center_to_mouth_center_height = (np.linalg.norm(f5 - c5) + np.linalg.norm(f5 - d5)) / 2

            features = [
                eye_height, eye_width, mouth_height, mouth_width,
                eyebrow_to_eye_center, eye_center_to_mouth_center_height
            ]

            return features

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

def plot_decision_boundary_with_input(X_train, y_train, X_test, y_test, model, input_point, prediction):
    plt.figure(figsize=(12, 8))
    # Decision boundary
    try:
        w = model.coef_[0]
        b = model.intercept_[0]
        x_points = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
        y_points = -(w[0] * x_points + b) / w[1]
        plt.plot(x_points, y_points, color='green', label='Decision Boundary')
    except Exception as e:
        print(f"Could not plot decision boundary: {e}")

    # Training data scatter
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='red', label='Sad')
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='blue', label='Happy')

    # New input data
    plt.scatter(input_point[0], input_point[1], color='yellow', label='Input', marker='x', s=100)
    emotion = "Happy" if prediction == 1 else "Sad"
    plt.title(f'Emotion Classification: {emotion}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Load training data
    base_path = 'Emotions Dataset'
    feature_extractor = EmotionFeatureExtractor()
    features, labels = [], []
    emotions = ['happy', 'sad']

    for emotion in emotions:
        path = os.path.join(base_path, 'test', emotion)
        label = 1 if emotion == 'happy' else 0
        for file in os.listdir(path):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(path, file)
                image = cv2.imread(img_path)
                feature = feature_extractor.calculate_emotion_features(image)
                if feature:
                    features.append(feature)
                    labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    # Preprocess data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    X_train, X_test, y_train, y_test = train_test_split(features_pca, labels, test_size=0.2, random_state=42)

    # Train SVM
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)

    # Capture and process input image
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture an image.")
    while True:
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the frame to match the training image dimensions (e.g., 300x300)
        gray_frame_resized = cv2.resize(gray_frame, (300, 300))

        # Display the resized grayscale frame
        cv2.imshow('Live Feed', gray_frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # Extract features from captured grayscale image
    input_features = feature_extractor.calculate_emotion_features(gray_frame_resized)
    if input_features is None:
        print("No face detected or unable to extract features.")
        return

    # Transform input features for prediction
    input_scaled = scaler.transform([input_features])
    input_pca = pca.transform(input_scaled)
    prediction = svm.predict(input_pca)[0]

    # Plot results
    plot_decision_boundary_with_input(X_train, y_train, X_test, y_test, svm, input_pca[0], prediction)


if __name__ == "__main__":
    main()
