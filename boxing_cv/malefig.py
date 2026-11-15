import cv2
from ultralytics import YOLO

model = YOLO('yolov8x-pose.pt')

# Укажите путь к любому клипу из папки clips/
video_path = "clips/01_bivol_smith_seg1_RUS-black_USA-white.mp4"
# Номер кадра, который хотим сохранить (например, 150-й кадр — это ~5-я секунда видео)
target_frame = 200

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
ret, frame = cap.read()

if ret:
    # Запускаем нейросеть на этом одном кадре
    results = model(frame, device=0)
    # Рисуем скелеты и рамки (labels=False и conf=False убирают лишний текст, оставляя чистые линии скелета)
    annotated_frame = list(results).__getitem__(0).plot(labels=False, boxes=False)


    cv2.imwrite("Figure1_skeleton.png", annotated_frame)
    print("Готово! Картинка сохранена как Figure1_skeleton.png")

cap.release()
