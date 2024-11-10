import cv2

# Initialize the camera
camera = cv2.VideoCapture(0)  # '0' is usually the built-in camera

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Read and display video frames
while True:
    ret, frame = camera.read()  # Read a frame
    if not ret:
        print("Error: Failed to grab the frame.")
        break

    # Display the frame
    cv2.imshow("Camera", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
