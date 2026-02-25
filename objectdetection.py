import cv2
from ultralytics import YOLO

# -----------------------------
# Load YOLOv8 Nano Model
# -----------------------------
# This will auto-download on first run
model = YOLO("yolov8n.pt")

# -----------------------------
# Raspberry Pi Stream URL
# -----------------------------
url = "http://10.226.121.78:5000/video_feed"

# Use FFMPEG backend for MJPEG stream
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Cannot open video stream. Check Pi server.")
    exit()

print("✅ Connected to Raspberry Pi stream")
print("🚀 Starting YOLO detection... Press 'q' to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to receive frame from stream")
        break

    # Resize for faster detection (important)
    frame = cv2.resize(frame, (640, 480))

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLOv8 Object Detection - Pi Stream", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




# import cv2
# from ultralytics import YOLO

# model = YOLO("yolov8s.pt")   # better than nano

# url = "http://10.226.121.78:5000/video_feed"
# cap = cv2.VideoCapture(url)

# if not cap.isOpened():
#     print("Cannot open stream")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (640, 640))

#     results = model(frame, conf=0.3, verbose=False)

#     annotated = results[0].plot()

#     cv2.imshow("YOLO Detection", annotated)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
