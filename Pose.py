import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# LOAD MODEL POSE LANDMARKER
base_options = python.BaseOptions(model_asset_path="dataset/pose_landmarker_heavy.task")
options = vision.PoseLandmarkerOptions(base_options=base_options)
detector = vision.PoseLandmarker.create_from_options(options)

# KONEKSI LANDMARK TUBUH
POSE_CONNECTIONS = [
    # Kepala
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),

    # Bahu
    (11,12),

    # Tangan kiri
    (11,13),
    (13,15),

    # Tangan kanan
    (12,14),
    (14,16),

    # Badan
    (11,23),
    (12,24),
    (23,24),

    # Kaki kiri
    (23,25),
    (25,27),
    (27,31),

    # Kaki kanan
    (24,26),
    (26,28),
    (28,32)
]


# OPEN CAMERA
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

    # DETEKSI POSE
    result = detector.detect(mp_image)
    if result.pose_landmarks:
        pose_landmarks = result.pose_landmarks[0]

        # GAMBAR GARIS SKELETON
        for start, end in POSE_CONNECTIONS:

            p1 = pose_landmarks[start]
            p2 = pose_landmarks[end]

            x1 = int(p1.x * w)
            y1 = int(p1.y * h)

            x2 = int(p2.x * w)
            y2 = int(p2.y * h)

            cv2.line(frame,(x1, y1),(x2, y2),(255, 0, 0),2)


        # GAMBAR LANDMARK
        for idx, lm in enumerate(pose_landmarks):

            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.circle(frame,(x, y),5,(0, 255, 0), -1)

            cv2.putText(frame,str(idx),(x + 5, y - 5),cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,(0, 0, 255),1)

    cv2.imshow("Pose Landmarker - MediaPipe Tasks", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
