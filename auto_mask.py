import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class SamSegmentation:
    """
    Класс SamSegmentation обеспечивает сегментацию изображений с использованием модели SAM (Segment Anything Model).
    """
    def __init__(self, model_type="vit_h", checkpoint_path="", device=None, min_area=500):
        """
        Инициализация класса SamSegmentation.

        :param model_type: Тип модели SAM для использования в сегментации.
        :type model_type: str
        :param checkpoint_path: Путь к файлу чекпойнта обученной модели.
        :type checkpoint_path: str
        :param device: Устройство для выполнения вычислений (например, 'cuda:0' или 'cpu').
        :type device: torch.device, optional
        :param min_area: Минимальная площадь области для учета в маске.
        :type min_area: int, optional
        """
        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.sam.to(device=self.device)
        print(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam, min_mask_region_area=min_area)

    # def segment_image(self, image):
    #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     masks = self.mask_generator.generate(image_rgb)
    #     return masks
    
    def segment_image(self, image):
        """
        Метод для сегментации изображения.

        :param image: Изображение для сегментации.
        :type image: np.ndarray
        :return: Список масок, где каждая маска представляет собой словарь с данными о сегментированной области.
        :rtype: list
        """
        height, width = image.shape[:2]
        bottom_half = image[height // 2:, :]
        image_rgb = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image_rgb)

        # Преобразуем маски обратно к размерам исходного изображения
        new_masks = []
        for mask in masks:
            mask_full_image = np.zeros((height, width), dtype=mask['segmentation'].dtype)
            mask_full_image[height // 2:, :] = mask['segmentation']
            new_mask = mask.copy()
            new_mask['segmentation'] = mask_full_image
            new_masks.append(new_mask)

        return new_masks