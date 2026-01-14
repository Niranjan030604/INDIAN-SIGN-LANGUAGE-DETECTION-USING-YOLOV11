# webcam_predict.py
import cv2
from ultralytics import YOLO
import time

# ------------------- CONFIG -------------------
MODEL_PATH = "runs/train/sign_lang_yolo11/weights/best.pt"   # <-- your trained model
CONFIDENCE = 0.4                                             # lower = more detections
WEBCAM_ID = 0                                                # 0 = default camera
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_COLOR = (255, 255, 255)   # White
BG_COLOR = (0, 100, 255)       # Orange background for text
# ---------------------------------------------

# Load the trained model
print(f"[INFO] Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(WEBCAM_ID)
if not cap.isOpened():
    raise IOError("Cannot open webcam! Check camera ID.")

print("[INFO] Starting live detection... Press 'q' to quit.")

# Warm-up inference (helps stabilize FPS)
for _ in range(5):
    ret, frame = cap.read()
    if ret:
        _ = model(frame, verbose=False)

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize for faster inference (optional)
    frame_resized = cv2.resize(frame, (640, 640))

    # Run inference
    results = model(frame_resized, conf=CONFIDENCE, verbose=False)[0]

    # Draw predictions
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        label = f"{model.names[cls_id]} {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label background
        (w, h), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 2)
        cv2.rectangle(frame_resized, (x1, y1 - h - 10), (x1 + w, y1), BG_COLOR, -1)

        # Draw label text
        cv2.putText(frame_resized, label, (x1, y1 - 5),
                    FONT, FONT_SCALE, FONT_COLOR, 2)

    # FPS counter
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame_resized, f"FPS: {fps:.1f}", (10, 30),
                FONT, 0.8, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Live Sign Language Detection", frame_resized)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam closed.")