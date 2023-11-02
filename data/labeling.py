import sys
sys.path.append("c:/Users/ivanin.em/Desktop/sam")
from auto_mask import SamSegmentation
from tools import DepthTools
import cv2
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def is_point_inside_bbox(point, bbox):
    """Проверяет, находится ли точка внутри ограничивающего прямоугольника."""
    x, y = point
    x1, y1, width, height = bbox
    return x1 <= x <= x1 + width and y1 <= y <= y1 + height

def is_point_inside_mask(point, mask):
    """Проверяет, находится ли точка внутри маски."""
    x, y = map(int, point)  # Преобразовываем координаты в целочисленный формат
    return mask[y, x]

MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = 'C:/Users/ivanin.em/Desktop/sam/sam_vit_h_4b8939.pth'
RGB_VIDEO_PATH = 'C:/Users/ivanin.em/Desktop/sam/rgb_output.avi'

cap_rgb = cv2.VideoCapture(RGB_VIDEO_PATH)
total_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
# total_frames = 20

segmenter = SamSegmentation(MODEL_TYPE, CHECKPOINT_PATH)

# Создаем структуру данных для CSV
data = {
    "area": [],
    "bbox_x": [],
    "bbox_y": [],
    "bbox_width": [],
    "bbox_height": [],
    "predicted_iou": [],
    "point_coords_x": [],
    "point_coords_y": [],
    "stability_score": [],
    "Y": []
}

with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
    for i in range(total_frames):
        ret_rgb, frame_rgb = cap_rgb.read()
        
        if not ret_rgb:
            break

        if i % 50 != 0:
            continue
        
        masks = segmenter.segment_image(frame_rgb)

        # Визуализация
        draw = DepthTools.draw_annotations(frame_rgb, masks)
        plt.imshow(draw)

        # clicked_mask = None
        plt.ion()  # Включаем интерактивный режим

        clicked_mask = []
        clicked_mask = plt.ginput(n=0)

        plt.ioff()  # Выключаем интерактивный режим


        # Итерируемся по маскам и записываем данные
        for mask in masks:
            bbox = mask['bbox']
            mask_points = mask['segmentation']

            data["area"].append(mask["area"])
            data["bbox_x"].append(mask['bbox'][0])
            data["bbox_y"].append(mask['bbox'][1])
            data["bbox_width"].append(mask['bbox'][2])
            data["bbox_height"].append(mask['bbox'][3])
            data["predicted_iou"].append(mask["predicted_iou"])
            data["point_coords_x"].append(mask["point_coords"][0][0])
            data["point_coords_y"].append(mask["point_coords"][0][1])
            data["stability_score"].append(mask["stability_score"])

            mask_clicked = 0
            for point in clicked_mask:
                if is_point_inside_mask(point, mask_points):
                    mask_clicked = 1
                    break

            data["Y"].append(mask_clicked)

        plt.close()
        pbar.update(1)

# Преобразуем данные в DataFrame и сохраняем в CSV
df = pd.DataFrame(data)
df.to_csv("masks_data.csv", index=False)

cap_rgb.release()