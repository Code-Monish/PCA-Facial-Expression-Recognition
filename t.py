<<<<<<< HEAD
# import cv2
# import numpy as np
# import time
# import matplotlib.pyplot as plt

# fig = plt.figure()
# cap = cv2.VideoCapture(0)  # Use 0 for the default camera
# pTime = 0
# frame_number = 0
# frame_numbers_list = []
# fps_list = []
# plt.ion()  # Turn on interactive mode for live plotting
# while True:
#     # Capture frame-by-frame
#     success, img = cap.read()
#     if not success:
#         break
    
#     img = cv2.resize(img, (640, 480))  # Resize for consistency
    
#     # Calculate FPS
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
    
#     # Append frame number and FPS to lists
#     frame_numbers_list.append(frame_number)
#     fps_list.append(int(fps))
#     frame_number += 1
    
#     # Plotting FPS data
#     plt.clf()  # Clear the current figure
#     plt.plot(frame_numbers_list, fps_list)
#     plt.title("FPS on every frame")
#     plt.ylabel("FPS")
#     plt.xlabel("Frame Number")
#     plt.pause(0.1)  # Pause to update the plot
#     # Convert Matplotlib figure to OpenCV image
#     fig.canvas.draw()
#     plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     plot = plot.reshape(fig.canvas.get_width_height())
#     plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
    
#     # Combine original frame and plot image
#     result_img = np.hstack([img, plot])
    
#     # Display the combined image
#     cv2.imshow("Camera Feed", result_img)
    
#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# # Release the capture and close windows
# cap.release()
# cv2.destroyAllWindows()

import cv2

cap = cv2.VideoCapture(0)  

while True:
    success, img = cap.read()
    if not success:
        break
    cv2.imshow("Camera Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
=======
>>>>>>> e44c27c (Progress)
