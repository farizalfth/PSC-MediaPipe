import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# LOAD MODEL FACE LANDMARKER
base_options = python.BaseOptions(
    model_asset_path="dataset/face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

detector = vision.FaceLandmarker.create_from_options(options)

# BEBERAPA KONEKSI WAJAH
FACE_CONNECTIONS = [
    # Mata kiri
    (33, 133),
    (133, 159),
    (159, 145),
    (145, 33),

    # Mata kanan
    (362, 263),
    (263, 386),
    (386, 374),
    (374, 362),

    # Bibir
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),

    # Hidung
    (1, 4),
    (4, 5),
    (5, 195)
]

# OPEN CAMERA
cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]

        # GAMBAR GARIS
        for start, end in FACE_CONNECTIONS:

            p1 = face_landmarks[start]
            p2 = face_landmarks[end]

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
        for idx, lm in enumerate(face_landmarks):

            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

        # TAMPILKAN NOMOR LANDMARK PENTING
        important_points = [
            1,          # hidung
            33, 133,    # mata kiri
            362, 263,   # mata kanan
            61, 291,    # bibir
            13, 14      # atas-bawah mulut
        ]

        for idx in important_points:

            lm = face_landmarks[idx]

            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.putText(
                frame,
                str(idx),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1
            )

    cv2.imshow("Face Mesh - MediaPipe Tasks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()