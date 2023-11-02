from auto_mask import SamSegmentation
from tools import DepthTools
import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
from tqdm import tqdm
import xgboost as xgb


# Загрузка модели
model = xgb.XGBClassifier()
model.load_model("data/best_model.xgb")


def extract_features(mask):
    bbox = mask['bbox']
    point_coords = mask['point_coords'][0] if mask['point_coords'] else [0, 0]
    features = [
        mask['area'],
        bbox[0], bbox[1], bbox[2], bbox[3],  # bbox_x, bbox_y, bbox_width, bbox_height
        mask['predicted_iou'],
        point_coords[0], point_coords[1],  # point_coords_x, point_coords_y
        mask['stability_score']
    ]
    return np.array(features).reshape(1, -1), mask['segmentation']

MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = 'C:/Users/ivanin.em/Desktop/sam/sam_vit_h_4b8939.pth'
RGB_VIDEO_PATH = 'C:/Users/ivanin.em/Desktop/sam/rgb_output.avi'
DEPTH_VIDEO_PATH = 'C:/Users/ivanin.em/Desktop/sam/depth_output.avi'
OUTPUT_VIDEO_PATH = 'C:/Users/ivanin.em/Desktop/sam/result.avi'

segmenter = SamSegmentation(MODEL_TYPE, CHECKPOINT_PATH)

frame_rgb = cv2.imread('frame_200.jpg')
frame_depth = cv2.imread('depth_frame_200.jpg')

masks = segmenter.segment_image(frame_rgb)
# filtered_masks = []
# best_threshold = 0.1
# for mask in masks:
#     features, segmentation = extract_features(mask)
#     prediction_proba = 1 - model.predict_proba(features)[:, 1]  # Получение вероятности принадлежности к классу 0
#     print(prediction_proba)
#     if prediction_proba > best_threshold:  # Использование порога классификации
#         filtered_masks.append(mask)


masks_list = [mask['segmentation'].astype(np.uint8) * 255 for mask in masks]  # Переводим маски в формат, пригодный для OpenCV
# masks_list = [mask['segmentation'].astype(np.uint8) * 255 for mask in filtered_masks]         

image_with_annotations = DepthTools.draw_annotations(frame_rgb, masks)

# Сторим и выводим карту препятсвий
avg_depth = DepthTools.calculate_average_depth(frame_depth, masks_list)
depth_img = DepthTools.draw_depth_map(frame_rgb, avg_depth, image_width=640, max_depth=6, show=False, save=True)

# Соединяем изображения и сохраняем в видеофайл
height, width, _ = image_with_annotations.shape
depth_img_resized = cv2.resize(depth_img, (width, height))
combined_img = np.hstack((image_with_annotations, depth_img_resized))
cv2.imshow('Combined Image', combined_img)
cv2.waitKey(0)
# cv2.imwrite('result_with_boost.jpg', combined_img)
