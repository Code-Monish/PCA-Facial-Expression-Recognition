face_cascade = cv2.CascadeClassifier('path_to/haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Could not load Haar Cascade.")
    exit()