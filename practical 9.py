import cv2
from ultralytics import YOLO


# Step 1: Load the pretrained YOLO model
model = YOLO("yolo11n.pt")


# Step 2: Open CCTV/video footage
video = cv2.VideoCapture("cctv.mp4")


# Check whether video opened successfully
if not video.isOpened():
    print("Error: Could not open video.")
    exit()


# Step 3: Process video frame by frame
while True:

    success, frame = video.read()

    # Stop when video ends
    if not success:
        break


    # Step 4: Perform object detection
    results = model(frame)


    # Step 5: Draw bounding boxes and labels
    annotated_frame = results[0].plot()


    # Step 6: Display the frame
    cv2.imshow("CCTV Object Detection", annotated_frame)


    # Press q to stop the video
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Step 7: Release resources
video.release()
cv2.destroyAllWindows()