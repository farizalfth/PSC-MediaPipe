import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# LOAD MODEL
base_options = python.BaseOptions(
    model_asset_path="dataset/hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

# HAND CONNECTIONS
connections = [
    (0,1),(1,2),(2,3),(3,4),          # Thumb
    (0,5),(5,6),(6,7),(7,8),          # Index
    (5,9),(9,10),(10,11),(11,12),     # Middle
    (9,13),(13,14),(14,15),(15,16),   # Ring
    (13,17),(17,18),(18,19),(19,20),  # Pinky
    (0,17)                            # Palm
]

# OPEN CAMERA
cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert ke MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Deteksi tangan
    result = detector.detect(mp_image)

    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            # GAMBAR GARIS
            for start, end in connections:

                p1 = hand_landmarks[start]
                p2 = hand_landmarks[end]

                x1 = int(p1.x * w)
                y1 = int(p1.y * h)

                x2 = int(p2.x * w)
                y2 = int(p2.y * h)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

            # GAMBAR LANDMARK
            for idx, lm in enumerate(hand_landmarks):

                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    str(idx),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1
                )

    cv2.imshow("Hand Landmarker - MediaPipe Tasks", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()