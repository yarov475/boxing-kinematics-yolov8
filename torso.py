import numpy as np


def calculate_torso_angle(keypoints):
    """
    keypoints[0]: нос, [11]: левое плечо, [12]: правое плечо,
                [23]: левый таз, [24]: правый таз
    """
    shoulder = (keypoints[11] + keypoints[12]) / 2  # Средняя точка плеч
    hip = (keypoints[23] + keypoints[24]) / 2  # Средняя точка таза

    # Вектор "плечи" -> горизонталь
    vec_shoulders = np.array([shoulder[0], 0])
    # Вектор "таз" -> "плечи"
    vec_torso = shoulder[:2] - hip[:2]

    cos_theta = np.dot(vec_torso, vec_shoulders) / (np.linalg.norm(vec_torso) * np.linalg.norm(vec_shoulders))
    angle = np.degrees(np.arccos(cos_theta))

    return round(angle, 1)