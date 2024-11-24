import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

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

def load_and_process_dataset(base_path):
    feature_extractor = EmotionFeatureExtractor()
    features_list = []
    labels = []

    for emotion, label in [('happy', 1), ('sad', 0)]:
        path = os.path.join(base_path, 'test', emotion)
        for filename in os.listdir(path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    features = feature_extractor.calculate_emotion_features(img)
                    if features is not None:
                        features_list.append(features)
                        labels.append(label)

    return np.array(features_list), np.array(labels)

def emotion_classification(base_path):
    # Load features
    features, labels = load_and_process_dataset(base_path)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features_pca, labels, test_size=0.2, random_state=42)
    
    # Train SVM
    svm = SVC(kernel='rbf')
    svm.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = svm.score(X_train, y_train)
    test_accuracy = svm.score(X_test, y_test)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(features_pca[labels == 1, 0], features_pca[labels == 1, 1], 
                color='blue', label='Happy')
    plt.scatter(features_pca[labels == 0, 0], features_pca[labels == 0, 1], 
                color='red', label='Sad')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Emotion Classification after PCA')
    plt.legend()
    
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    
    plt.show()

def main():
    base_path = 'Emotions Dataset'
    emotion_classification(base_path)

if __name__ == "__main__":
    main()