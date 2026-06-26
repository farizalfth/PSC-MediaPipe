from socket import J1939_NLA_BYTES_ACKED

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================
# LOAD MODEL
# ==========================
base_options = python.BaseOptions(
    model_asset_path="dataset/hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# ==========================
# COUNTER
# ==========================
count = 0
crossed_right = False
J1939_NLA_BYTES_ACKED
# ==========================
# CAMERA
# ==========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Webcam tidak dapat dibuka!")
    exit()

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame kamera")
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Garis batas
    left_line = int(w * 0.3)
    right_line = int(w * 0.7)

    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    # ==========================
    # DETEKSI TANGAN
    # ==========================
    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]

        # Wrist = landmark 0
        wrist = hand_landmarks[0]

        cx = int(wrist.x * w)
        cy = int(wrist.y * h)

        # Titik wrist
        cv2.circle(
            frame,
            (cx, cy),
            10,
            (0, 255, 0),
            -1
        )

        # Tampilkan koordinat
        cv2.putText(
            frame,
            f"X: {cx}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # ==========================
        # TRACKING KANAN -> KIRI
        # ==========================

        # Sudah melewati START
        if cx > right_line:
            crossed_right = True

        # Lalu melewati FINISH
        if crossed_right and cx < left_line:
            count += 1
            crossed_right = False

    # ==========================
    # VISUALISASI
    # ==========================

    # Garis finish
    cv2.line(
        frame,
        (left_line, 0),
        (left_line, h),
        (0, 255, 0),
        2
    )

    # Garis start
    cv2.line(
        frame,
        (right_line, 0),
        (right_line, h),
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "FINISH",
        (left_line - 50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "START",
        (right_line - 40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"RIGHT -> LEFT : {count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        3
    )

    # ==========================
    # TAMPILKAN WINDOW
    # ==========================
    cv2.imshow("Hand Gesture Counter", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ==========================
# CLEANUP
# ==========================
cap.release()
cv2.destroyAllWindows()