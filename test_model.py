from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# HAND
hand_base = python.BaseOptions(
    model_asset_path="dataset/hand_landmarker.task"
)

hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base,
    num_hands=1
)

hand_detector = vision.HandLandmarker.create_from_options(hand_options)

print("✅ Hand Model OK")

# FACE
face_base = python.BaseOptions(
    model_asset_path="dataset/face_landmarker.task"
)

face_options = vision.FaceLandmarkerOptions(
    base_options=face_base,
    num_faces=1
)

face_detector = vision.FaceLandmarker.create_from_options(face_options)

print("✅ Face Model OK")

# POSE
pose_base = python.BaseOptions(
    model_asset_path="dataset/pose_landmarker_heavy.task"
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base
)

pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

print("✅ Pose Model OK")

print("\n🎉 Semua model berhasil dimuat!")