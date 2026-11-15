import os
import subprocess
import glob
import pandas as pd
from scipy.stats import mannwhitneyu
# Импортируем функцию анализа из вашего предыдущего файла batch_analyze_color.py
from batch_analyze_color import process_clip_by_color

os.makedirs("clips", exist_ok=True)

# 1. НАСТРОЙКА БОЕВ И ТАЙМКОДОВ ДЛЯ АВТОМАТИЧЕСКОЙ НАРЕЗКИ
FIGHTS = [
    {
        "name": "01_bivol_smith",
        "url": "https://vkvideo.ru/video-172995863_456239029",
        "rus_color": "black",  # Черные шорты Бивола
        "usa_color": "white",  # Белые шорты Смита
        "segments": [
            ("04:10", "04:45"),  # 2-й раунд (общий план)
            ("12:15", "12:50"),  # 4-й раунд
            ("25:30", "26:05"),  # 8-й раунд
        ]
    },
    {
        "name": "02_kovalev_ward",
        "url": "https://vkvideo.ru/video-20701067_456240660",
        "rus_color": "blue",  # Проверьте цвет шорт в вашем видео
        "usa_color": "white",
        "segments": [
            ("06:20", "06:55"),
            ("15:10", "15:45"),
            ("28:00", "28:35"),
        ]
    },
    # Добавьте еще 2-3 боя по этому же шаблону
]


def download_and_cut():
    for fight in FIGHTS:
        if "ВСТАВЬТЕ" in fight["url"]:
            continue
        for idx, (start, end) in enumerate(fight["segments"], 1):
            fname = f"clips/{fight['name']}_seg{idx}_RUS-{fight['rus_color']}_USA-{fight['usa_color']}.mp4"
            if os.path.exists(fname):
                print(f"Уже скачано: {fname}")
                continue

            print(f"Скачиваю и нарезаю {fname} ({start} - {end})...")
            cmd = [
                "yt-dlp",
                "--download-sections", f"*{start}-{end}",
                "-f", "bestvideo[height<=1080][ext=mp4]",
                "--force-keyframes-at-cuts",
                "-o", fname,
                fight["url"]
            ]
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # Этап 1: Автоматическая нарезка с YouTube
    download_and_cut()

    # Этап 2: Анализ всех клипов на видеокарте
    all_results = []
    for vf in glob.glob("clips/*.mp4"):
        fname = os.path.basename(vf).lower().replace('.mp4', '')
        parts = fname.split('_')
        rus_c = next((p.split('-') for p in parts if p.startswith('rus-')), None)
        usa_c = next((p.split('-') for p in parts if p.startswith('usa-')), None)
        if rus_c and usa_c:
            print(f"Нейросеть анализирует: {vf}...")
            all_results.extend(process_clip_by_color(vf, rus_c, usa_c))

    if all_results:
        df_final = pd.DataFrame(all_results)
        df_final.to_csv("final_scopus_metrics.csv", index=False)

        print("\n=== ГОТОВЫЕ ЦИФРЫ ДЛЯ СТАТЬИ (SCOPUS) ===")
        rus = df_final[df_final['school'] == 'POST_SOVIET']
        usa = df_final[df_final['school'] == 'USA']
        for m in ['median_distance', 'pendulum_freq_hz', 'dist_velocity', 'median_torso_angle', 'torso_evasion_iqr',
                  'torso_angular_vel']:
            stat, p = mannwhitneyu(rus[m], usa[m])
            print(f"{m:20s} | RUS: {rus[m].median():.3f} (IQR {rus[m].quantile(0.75) - rus[m].quantile(0.25):.3f}) "
                  f"| USA: {usa[m].median():.3f} (IQR {usa[m].quantile(0.75) - usa[m].quantile(0.25):.3f}) | p = {p:.4e}")
