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
# FOLDER
# ======================================================
input_folder = "dataset/gambar"
output_folder = "dataset/hasil"

os.makedirs(output_folder, exist_ok=True)

# ======================================================
# KONEKSI LANDMARK TANGAN
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
# MEMBUAT CSV
# ======================================================
with open("hand_landmark.csv", "w", newline="") as csv_file:

    writer = csv.writer(csv_file)

    header = ["gambar"]

    for i in range(21):
        header.extend([
            f"x_{i}",
            f"y_{i}",
            f"z_{i}"
        ])

    writer.writerow(header)

    print("=" * 50)
    print("Mulai memproses dataset...")
    print("=" * 50)

    # ==================================================
    # PROSES SEMUA GAMBAR
    # ==================================================
    for nama_file in sorted(os.listdir(input_folder)):

        if not nama_file.lower().startswith("tangan"):
            continue

        image_path = os.path.join(input_folder, nama_file)

        # MediaPipe Image
        mp_image = mp.Image.create_from_file(image_path)

        # Deteksi
        result = detector.detect(mp_image)

        # OpenCV Image
        img = cv2.imread(image_path)

        if img is None:
            print(f"Gagal membuka {nama_file}")
            continue

        h, w, _ = img.shape

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # ==========================
            # SIMPAN KE CSV
            # ==========================
            row = [nama_file]

            for lm in hand:
                row.extend([lm.x, lm.y, lm.z])

            writer.writerow(row)

            # ==========================
            # GAMBAR GARIS
            # ==========================
            points = []

            for lm in hand:
                px = int(lm.x * w)
                py = int(lm.y * h)
                points.append((px, py))

            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    img,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )

            # ==========================
            # GAMBAR TITIK
            # ==========================
            for point in points:
                cv2.circle(
                    img,
                    point,
                    5,
                    (0, 0, 255),
                    -1
                )

            # ==========================
            # SIMPAN HASIL
            # ==========================
            output_path = os.path.join(
                output_folder,
                f"{os.path.splitext(nama_file)[0]}_result.jpg"
            )

            cv2.imwrite(output_path, img)

            # ==========================
            # TAMPILKAN
            # ==========================
            cv2.imshow("Hand Landmark Detection", img)
            cv2.waitKey(0)

            print(f"✓ {nama_file} berhasil diproses")
            print(f"  -> {output_path}")

        else:
            print(f"✗ Tidak ada tangan pada {nama_file}")

cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("SELESAI")
print("CSV        : hand_landmark.csv")
print(f"Hasil Gambar : {output_folder}")
print("=" * 50)