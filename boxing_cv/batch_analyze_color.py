import os
import glob
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import mannwhitneyu
from ultralytics import YOLO

model = YOLO('yolov8x-pose.pt')


def get_pt(kp_array, idx):
    """Безопасно извлекает координаты (x, y) точки по индексу."""
    row = kp_array.tolist()
    return float(row[idx].__getitem__(0)), float(row[idx].__getitem__(1))


def get_shorts_color_score(frame_hsv, kp_array, target_color):
    """Оценивает совпадение цвета в зоне бедер-коленей (шорт)."""
    pts = [get_pt(kp_array, i) for i in (11, 12, 13, 14)]
    for x, y in pts:
        if x == 0.0 or y == 0.0:
            return -1.0

    xs = [p.__getitem__(0) for p in pts]
    ys = [p.__getitem__(1) for p in pts]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    pad_x = int((x_max - x_min) * 0.15)
    img_h, img_w = int(frame_hsv.shape.__getitem__(0)), int(frame_hsv.shape.__getitem__(1))

    x_min = max(0, x_min + pad_x)
    x_max = min(img_w, x_max - pad_x)
    y_min = max(0, y_min)
    y_max = min(img_h, y_max)

    if x_max <= x_min or y_max <= y_min:
        return -1.0

    roi = np.asarray(frame_hsv)[y_min:y_max, x_min:x_max]
    h_chan, s_chan, v_chan = cv2.split(roi)

    if target_color == 'white':
        mask = (s_chan < 60) & (v_chan > 160)
    elif target_color == 'black':
        mask = (v_chan < 65)
    elif target_color == 'red':
        mask = ((h_chan < 12) | (h_chan > 168)) & (s_chan > 90) & (v_chan > 60)
    elif target_color == 'blue':
        mask = (h_chan >= 95) & (h_chan <= 135) & (s_chan > 80) & (v_chan > 50)
    else:
        return 0.0

    return float(np.mean(mask))


def calculate_angle(p1, p2):
    """Вычисляет угол наклона корпуса (в градусах) относительно горизонтали."""
    dx = float(p2.__getitem__(0)) - float(p1.__getitem__(0))
    dy = float(p2.__getitem__(1)) - float(p1.__getitem__(1))
    return float(np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-6)))


def process_clip_by_color(video_path, rus_color, usa_color):
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    effective_fps = fps / 2.0

    records = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 2 != 0:
            continue

        res_list = list(model(frame, verbose=False, device=0))
        if not res_list:
            continue
        res = res_list.__getitem__(0)

        if res.boxes is None or res.keypoints is None or len(res.boxes) < 2:
            continue

        areas = []
        for i, box in enumerate(res.boxes.xywh):
            b_vals = box.tolist()
            areas.append((float(b_vals.__getitem__(2)) * float(b_vals.__getitem__(3)), i))
        areas.sort(reverse=True)

        idx1 = int(areas.__getitem__(0).__getitem__(1))
        idx2 = int(areas.__getitem__(1).__getitem__(1))

        kp1 = res.keypoints.xy.__getitem__(idx1).cpu().numpy()
        kp2 = res.keypoints.xy.__getitem__(idx2).cpu().numpy()

        valid = True
        for pt_i in (5, 6, 11, 12):
            x1, y1 = get_pt(kp1, pt_i)
            x2, y2 = get_pt(kp2, pt_i)
            if x1 == 0.0 or y1 == 0.0 or x2 == 0.0 or y2 == 0.0:
                valid = False
                break
        if not valid:
            continue

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        s1_rus = get_shorts_color_score(frame_hsv, kp1, rus_color)
        s2_rus = get_shorts_color_score(frame_hsv, kp2, rus_color)
        s1_usa = get_shorts_color_score(frame_hsv, kp1, usa_color)
        s2_usa = get_shorts_color_score(frame_hsv, kp2, usa_color)

        if (s1_rus + s2_usa) > (s2_rus + s1_usa):
            rus_kp, usa_kp = kp1, kp2
        else:
            rus_kp, usa_kp = kp2, kp1

        rus_hip = (np.array(get_pt(rus_kp, 11)) + np.array(get_pt(rus_kp, 12))) / 2.0
        usa_hip = (np.array(get_pt(usa_kp, 11)) + np.array(get_pt(usa_kp, 12))) / 2.0
        rus_sh = (np.array(get_pt(rus_kp, 5)) + np.array(get_pt(rus_kp, 6))) / 2.0
        usa_sh = (np.array(get_pt(usa_kp, 5)) + np.array(get_pt(usa_kp, 6))) / 2.0

        rus_torso = float(np.linalg.norm(rus_sh - rus_hip))
        usa_torso = float(np.linalg.norm(usa_sh - usa_hip))
        if rus_torso < 20.0 or usa_torso < 20.0:
            continue

        # Нормированная координата X центра масс каждого бойца (индивидуальное движение)
        rus_norm_x = float(rus_hip.__getitem__(0)) / rus_torso
        usa_norm_x = float(usa_hip.__getitem__(0)) / usa_torso

        records.append({
            'frame': frame_idx,
            'school': 'POST_SOVIET',
            'norm_x': rus_norm_x,
            'torso_angle': calculate_angle(rus_hip, rus_sh)
        })
        records.append({
            'frame': frame_idx,
            'school': 'USA',
            'norm_x': usa_norm_x,
            'torso_angle': calculate_angle(usa_hip, usa_sh)
        })

    cap.release()
    df_raw = pd.DataFrame(records)
    if df_raw.empty:
        return []

    clip_summaries = []
    for school_name in ('POST_SOVIET', 'USA'):
        sub = df_raw[df_raw['school'] == school_name].sort_values('frame')
        if len(sub) < effective_fps * 4:
            continue

        win = max(5, int(effective_fps * 0.3) | 1)
        x_s = savgol_filter(sub['norm_x'].to_numpy(), win, 2)
        ang_s = savgol_filter(sub['torso_angle'].to_numpy(), win, 2)

        # Пики горизонтального смещения = индивидуальный «челнок» (смена направления движения)
        peaks, _ = find_peaks(x_s, prominence=0.10, distance=int(effective_fps * 0.35))
        troughs, _ = find_peaks(-x_s, prominence=0.10, distance=int(effective_fps * 0.35))
        duration = float(len(sub)) / effective_fps

        clip_summaries.append({
            'clip': os.path.basename(video_path),
            'school': school_name,
            'pendulum_freq_hz': float(len(peaks) + len(troughs)) / duration,
            'linear_mobility_vel': float(np.mean(np.abs(np.gradient(x_s, 1.0 / effective_fps)))),
            'median_torso_angle': float(np.median(ang_s)),
            'torso_evasion_iqr': float(np.percentile(ang_s, 75) - np.percentile(ang_s, 25)),
            'torso_angular_vel': float(np.mean(np.abs(np.gradient(ang_s, 1.0 / effective_fps))))
        })

    return clip_summaries



if __name__ == "__main__":
    video_files = glob.glob("clips/*.mp4")
    all_results = []

    for vf in video_files:
        file_title = os.path.basename(vf).lower().replace('.mp4', '')
        parts = file_title.split('_')
        rus_c = next((p.split('-').__getitem__(1) for p in parts if p.startswith('rus-')), None)
        usa_c = next((p.split('-').__getitem__(1) for p in parts if p.startswith('usa-')), None)

        if not rus_c or not usa_c:
            print(f"Пропущен {vf}: укажите цвета в имени, например RUS-black_USA-white.mp4")
            continue

        print(f"Анализ {vf} (RUS: {rus_c}, USA: {usa_c})...")
        all_results.extend(process_clip_by_color(vf, rus_c, usa_c))

    if all_results:
        df_final = pd.DataFrame(all_results)
        df_final.to_csv("final_scopus_metrics.csv", index=False)

        print("\n=== ИТОГОВАЯ СТАТИСТИКА ДЛЯ РАЗДЕЛА RESULTS ===")
        rus = df_final[df_final['school'] == 'POST_SOVIET']
        usa = df_final[df_final['school'] == 'USA']

        metrics = ('pendulum_freq_hz', 'linear_mobility_vel',
                   'median_torso_angle', 'torso_evasion_iqr', 'torso_angular_vel')

        for m in metrics:
            stat, p = mannwhitneyu(rus[m], usa[m])
            print(f"{m:20s} | RUS: {rus[m].median():.3f} (IQR {rus[m].quantile(0.75) - rus[m].quantile(0.25):.3f}) "
                  f"| USA: {usa[m].median():.3f} (IQR {usa[m].quantile(0.75) - usa[m].quantile(0.25):.3f}) | p = {p:.4e}")
    else:
        print("Не удалось собрать статистику. Проверьте, есть ли файлы в папке clips/ и видны ли боксеры в полный рост.")
