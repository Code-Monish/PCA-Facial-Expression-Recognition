import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

        mouth_left = np.array([landmarks[11].x, landmarks[11].y])
        mouth_right = np.array([landmarks[12].x, landmarks[12].y])
        mouth_center = (mouth_left + mouth_right) / 2

        left_eyebrow_top = np.array([landmarks[105].x, landmarks[105].y])
        right_eyebrow_top = np.array([landmarks[334].x, landmarks[334].y])
        eyebrow_height = np.mean([left_eyebrow_top[1], right_eyebrow_top[1]])

        left_eye_top = np.array([landmarks[159].x, landmarks[159].y])
        left_eye_bottom = np.array([landmarks[145].x, landmarks[145].y])
        eye_openness = np.linalg.norm(left_eye_top - left_eye_bottom)

        mouth_width = np.linalg.norm(mouth_left - mouth_right)

        def calculate_mouth_curvature(left, center, right):
            vec1 = left - center
            vec2 = right - center
            return np.degrees(np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))

        mouth_curvature = calculate_mouth_curvature(mouth_left, mouth_center, mouth_right)

        nose_left = np.array([landmarks[131].x, landmarks[131].y])
        nose_right = np.array([landmarks[359].x, landmarks[359].y])
        nose_width = np.linalg.norm(nose_left - nose_right)

        return [
            mouth_center[0], mouth_center[1], eyebrow_height, eye_openness,
            mouth_width, mouth_curvature, nose_width
        ]

class EmotionDetector:
    def __init__(self, base_path):
        self.base_path = base_path
        self.feature_extractor = EmotionFeatureExtractor()
        self.classifier = SVC(kernel='linear')

    def load_dataset(self):
        happy_path = os.path.join(self.base_path, 'test/happy')
        sad_path = os.path.join(self.base_path, 'test/sad')

        features_list = []
        labels = []

        for filename in os.listdir(happy_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(happy_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    features = self.feature_extractor.calculate_emotion_features(img)
                    if features is not None:
                        features_list.append(features)
                        labels.append(1)

        for filename in os.listdir(sad_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(sad_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    features = self.feature_extractor.calculate_emotion_features(img)
                    if features is not None:
                        features_list.append(features)
                        labels.append(0)

        return np.array(features_list), np.array(labels)

    def train_classifier(self):
        features, labels = self.load_dataset()
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    def detect_emotion(self, frame):
        features = self.feature_extractor.calculate_emotion_features(frame)
        if features is not None:
            prediction = self.classifier.predict([features])
            return "Happy" if prediction == 1 else "Sad"
        return "No face detected"

def main():
    base_path = 'Emotions Dataset'
    detector = EmotionDetector(base_path)
    detector.train_classifier()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion = detector.detect_emotion(frame)
        cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Emotion Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()