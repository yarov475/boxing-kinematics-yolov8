import os
import glob
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import mannwhitneyu
from ultralytics import YOLO

model = YOLO('yolov8x-pose.pt')


def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-6))


def process_single_clip(video_path, left_school, right_school):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    effective_fps = fps / 2.0  # так как берем каждый 2-й кадр

    records = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 2 != 0:
            continue

        results = model(frame, verbose=False, device=0)[0]
        boxes = results.boxes
        keypoints = results.keypoints

        if boxes is None or len(boxes) < 2:
            continue

        # Берем двух самых крупных людей в кадре
        areas = [(box[2].item() * box[3].item(), i) for i, box in enumerate(boxes.xywh)]
        areas.sort(reverse=True)
        idx1, idx2 = areas[0][1], areas[1][1]

        kp1 = keypoints.xy[idx1].cpu().numpy()
        kp2 = keypoints.xy[idx2].cpu().numpy()

        if np.any(kp1[[5, 6, 11, 12]] == 0) or np.any(kp2[[5, 6, 11, 12]] == 0):
            continue

        hip1_x = (kp1[11][0] + kp1[12][0]) / 2
        hip2_x = (kp2[11][0] + kp2[12][0]) / 2

        left_kp, right_kp = (kp1, kp2) if hip1_x < hip2_x else (kp2, kp1)

        left_hip = (left_kp[11] + left_kp[12]) / 2
        right_hip = (right_kp[11] + right_kp[12]) / 2
        left_sh = (left_kp[5] + left_kp[6]) / 2
        right_sh = (right_kp[5] + right_kp[6]) / 2

        avg_torso = (np.linalg.norm(left_sh - left_hip) + np.linalg.norm(right_sh - right_hip)) / 2
        if avg_torso < 20:
            continue

        norm_dist = np.linalg.norm(left_hip - right_hip) / avg_torso
        if norm_dist < 0.3 or norm_dist > 4.5:  # отсекаем глубокий клинч и дальний разрыв
            continue

        records.append({
            'frame': frame_idx,
            'school': left_school,
            'norm_distance': norm_dist,
            'torso_angle': calculate_angle(left_hip, left_sh)
        })
        records.append({
            'frame': frame_idx,
            'school': right_school,
            'norm_distance': norm_dist,
            'torso_angle': calculate_angle(right_hip, right_sh)
        })

    cap.release()
    df_raw = pd.DataFrame(records)
    if df_raw.empty:
        return []

    # Агрегируем метрики с фильтрацией Савицкого — Голея для каждого боксера в клипе
    clip_summaries = []
    for school_name in [left_school, right_school]:
        sub = df_raw[df_raw['school'] == school_name].sort_values('frame')
        if len(sub) < effective_fps * 4:  # минимум 4 секунды чистого трекинга
            continue

        win = int(effective_fps * 0.3)
        if win % 2 == 0: win += 1
        win = max(win, 5)

        dist_s = savgol_filter(sub['norm_distance'].values, win, 2)
        ang_s = savgol_filter(sub['torso_angle'].values, win, 2)

        peaks, _ = find_peaks(dist_s, prominence=0.12, distance=int(effective_fps * 0.35))
        duration = len(sub) / effective_fps

        clip_summaries.append({
            'clip': os.path.basename(video_path),
            'school': school_name,
            'median_distance': np.median(dist_s),
            'pendulum_freq_hz': len(peaks) / duration,
            'dist_velocity': np.mean(np.abs(np.gradient(dist_s, 1.0 / effective_fps))),
            'median_torso_angle': np.median(ang_s),
            'torso_evasion_iqr': np.percentile(ang_s, 75) - np.percentile(ang_s, 25),
            'torso_angular_vel': np.mean(np.abs(np.gradient(ang_s, 1.0 / effective_fps)))
        })

    return clip_summaries


if __name__ == "__main__":
    os.makedirs("clips", exist_ok=True)
    video_files = glob.glob("clips/*.mp4")

    if not video_files:
        print("Положите видео в папку clips/ с именами вида: fight1_RUS_USA.mp4 или fight2_USA_RUS.mp4")
    else:
        all_results = []
        for vf in video_files:
            fname = os.path.basename(vf).upper()
            # Определяем порядок по имени файла
            if "RUS_USA" in fname:
                l_sch, r_sch = "POST_SOVIET", "USA"
            elif "USA_RUS" in fname:
                l_sch, r_sch = "USA", "POST_SOVIET"
            else:
                print(f"Пропущен файл {vf}: в имени должно быть RUS_USA или USA_RUS")
                continue

            print(f"Обработка: {vf} ({l_sch} слева, {r_sch} справа)...")
            all_results.extend(process_single_clip(vf, l_sch, r_sch))

        df_final = pd.DataFrame(all_results)
        df_final.to_csv("final_scopus_metrics.csv", index=False)

        print("\n=== ИТОГОВАЯ СТАТИСТИКА ДЛЯ РАЗДЕЛА RESULTS ===")
        rus = df_final[df_final['school'] == 'POST_SOVIET']
        usa = df_final[df_final['school'] == 'USA']

        metrics = ['median_distance', 'pendulum_freq_hz', 'dist_velocity',
                   'median_torso_angle', 'torso_evasion_iqr', 'torso_angular_vel']

        for m in metrics:
            stat, p = mannwhitneyu(rus[m], usa[m])
            print(f"{m:20s} | RUS: {rus[m].median():.3f} (IQR {rus[m].quantile(0.75) - rus[m].quantile(0.25):.3f}) "
                  f"| USA: {usa[m].median():.3f} (IQR {usa[m].quantile(0.75) - usa[m].quantile(0.25):.3f}) | p = {p:.4e}")
