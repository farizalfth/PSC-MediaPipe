import os
import csv
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ======================================================
# LOAD MODEL
# ======================================================
base_options = python.BaseOptions(
    model_asset_path="dataset/hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# ======================================================
# PATH
# ======================================================
video_path = "dataset/video/hand.mp4"
output_folder = "dataset/hasil"

os.makedirs(output_folder, exist_ok=True)

output_video = os.path.join(output_folder, "hand_result.mp4")

# ======================================================
# HAND CONNECTIONS
# ======================================================
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ======================================================
# VIDEO
# ======================================================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video tidak ditemukan.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

writer_video = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

# ======================================================
# CSV
# ======================================================
csv_folder = "hasil_csv"
os.makedirs(csv_folder, exist_ok=True)

csv_path = os.path.join(csv_folder, "video_hand_landmark.csv")

with open(csv_path, "w", newline="") as csv_file:

    writer = csv.writer(csv_file)

    header = ["frame"]

    for i in range(21):
        header.extend([
            f"x_{i}",
            f"y_{i}",
            f"z_{i}"
        ])

    writer.writerow(header)

    frame_number = 0

    print("Memproses video...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # ==========================
            # Simpan CSV
            # ==========================
            row = [frame_number]

            for lm in hand:
                row.extend([lm.x, lm.y, lm.z])

            writer.writerow(row)

            # ==========================
            # Titik pixel
            # ==========================
            points = []

            for lm in hand:
                px = int(lm.x * width)
                py = int(lm.y * height)
                points.append((px, py))

            # ==========================
            # Gambar garis
            # ==========================
            for start, end in HAND_CONNECTIONS:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0,255,0),
                    2
                )

            # ==========================
            # Gambar titik
            # ==========================
            for p in points:

                cv2.circle(
                    frame,
                    p,
                    4,
                    (0,0,255),
                    -1
                )

        # ==========================
        # Simpan video
        # ==========================
        writer_video.write(frame)

        # ==========================
        # Resize tampilan
        # ==========================
        display = cv2.resize(frame, (700, 450))

        cv2.imshow("Hand Video Detection", display)

        frame_number += 1

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

cap.release()
writer_video.release()
cv2.destroyAllWindows()

print("="*50)
print("SELESAI")
print(f"CSV      : {csv_path}")
print(f"VIDEO    : {output_video}")
print("="*50)