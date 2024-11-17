import cv2
from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get(Models.YOLO_NAS_S, num_classes = 24, checkpoint_path="/home/om/Documents/NEW_VS/projects/ASL Recognization/Model/ckpt_latest.pth")

output = model.predict_webcam()

# # Open the webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()

#     # Perform object detection on the frame
#     results = model.predict(frame)

#     prediction = results.prediction[0] 

#     # Draw bounding boxes around the detected objects
#     for box in prediction.bboxes_xyxy:
#         x1, y1, x2, y2 = box.xyxy[0]
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

#     # Display the frame
#     cv2.imshow("Webcam", frame)

#     # Exit if the user presses the 'q' key
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Release the webcam and destroy all windows
# cap.release()
# cv2.destroyAllWindows()