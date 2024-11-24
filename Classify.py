import numpy as np
import cv2
import os
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class EmotionFeatureExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1, 
            min_detection_confidence=0.5
        )

    def calculate_emotion_features(self, image):
        # Resize image for consistent processing
        image = cv2.resize(image, (300, 300))
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.face_mesh.process(rgb_image)

        # Check if face is detected
        if not results.multi_face_landmarks:
            return None

        # Get first face landmarks
        landmarks = results.multi_face_landmarks[0].landmark

        # Ensure sufficient landmarks are detected
        if len(landmarks) < 468:
            print(f"Insufficient landmarks: {len(landmarks)}")
            return None

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
                return np.array([landmarks[idx].x, landmarks[idx].y]) if idx is not None else None

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

def load_dataset(base_path):
    feature_extractor = EmotionFeatureExtractor()
    features_list = []
    labels = []
    emotions = ['happy', 'sad']  # Expanded emotion set

    for emotion in emotions:
        path = os.path.join(base_path, 'test', emotion)
        label = emotions.index(emotion)
        
        for filename in os.listdir(path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(path, filename)
                
                try:
                    img = cv2.imread(img_path)
                    features = feature_extractor.calculate_emotion_features(img)
                    
                    if features is not None:
                        features_list.append(features)
                        labels.append(label)
                
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    return np.array(features_list), np.array(labels)

def emotion_classification(base_path):
    # Load and preprocess data
    features, labels = load_dataset(base_path)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Dimensionality reduction
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_pca, labels, test_size=0.2, random_state=42
    )
    
    # Kernels to test
    kernels = ['linear', 'rbf', 'poly']
    
    for kernel in kernels:
        # Train SVM
        svm = SVC(kernel=kernel, probability=True)
        svm.fit(X_train, y_train)
        
        # Predictions
        y_pred = svm.predict(X_test)
        
        # Evaluation
        print(f"\n{kernel.upper()} Kernel Performance:")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {kernel.upper()} Kernel')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

def main():
    base_path = 'Emotions Dataset'
    emotion_classification(base_path)

if __name__ == "__main__":
    main()