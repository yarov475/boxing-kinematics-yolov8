# Kinematic Analysis of Boxing Schools via Computer Vision (YOLOv8-pose)

This repository contains the source code, kinematic feature extraction pipeline, and aggregated statistical dataset accompanying the research paper:
**"Comparative Kinematic Analysis of Russian and American Boxing Techniques Using the YOLOv8-pose Model"**.

## Overview

The project implements an automated, markerless computer vision pipeline to quantify biomechanical and tactical differences between the Post-Soviet (Russian) and American boxing schools in professional bouts:
1. **Pose Estimation:** Extracts 17 COCO anatomical keypoints per athlete using `YOLOv8x-pose` (Ultralytics) with CUDA acceleration.
2. **Athlete Re-identification:** Tracks fighters across frames via HSV color histogram analysis in the anatomical hip-to-knee region (boxing trunks).
3. **Signal Processing:** Applies Savitzky–Golay polynomial smoothing (0.3 s window, 2nd order) and normalizes spatial displacements by individual torso length ($L_{\text{torso}}$) to eliminate camera zoom artifacts.
4. **Biomechanical Metrics:** Computes linear center-of-mass mobility ($V_{\text{lin}}$), pendulum step frequency ($F_{\text{pend}}$), baseline torso inclination ($\alpha_{\text{torso}}$), postural evasion amplitude ($\text{IQR}_{\text{angle}}$), and torso angular velocity ($\omega_{\text{torso}}$).
5. **Statistical Analysis:** Evaluates group differences using the non-parametric Mann–Whitney U test (`SciPy`).

## Repository Structure

* `auto_cut.py` — Automated video segment extraction utility (`yt-dlp` + `ffmpeg`).
* `batch_analyze_color.py` — Main pose estimation, HSV re-identification, and kinematic metric calculation pipeline.
* `make_final_plot.py` — Generation of high-resolution (300 DPI) distribution plots for publication.
* `final_scopus_metrics.csv` — Aggregated clip-level kinematic dataset used for statistical testing *(optional: remove `*.csv` from `.gitignore` if you want to publish the resulting table)*.

## Requirements & Setup

* Python 3.10+
* PyTorch with CUDA support
* Dependencies:
```bash
pip install ultralytics opencv-python pandas numpy scipy seaborn matplotlib
```

## Usage

1. Place cropped bout video clips (30–35 s, wide-angle camera view) into the `clips/` directory using the naming convention: `fightID_RUS-<trunks_color>_USA-<trunks_color>.mp4` (e.g., `01_bivol_smith_RUS-black_USA-white.mp4`).
2. Run the kinematic extraction pipeline:
```bash
python batch_analyze_color.py
```
3. Generate statistical summaries and figures:
```bash
python make_final_plot.py
```

## Citation & Data Availability

If you use this code or methodology in your research, please cite the corresponding paper. Raw broadcast video files are excluded due to copyright restrictions; however, exact bout timestamps and extracted kinematic metrics are provided for full reproducibility.
