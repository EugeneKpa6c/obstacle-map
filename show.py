from auto_mask import SamSegmentation
from tools import DepthTools
import numpy as np
import cv2

class ShowResult:
    """    
    Предоставляет функциональность для объединения результатов сегментации изображения и анализа данных глубины.
    """

    def __init__(self):
        """
        Инициализация модели сегментации изображений.

        Атрибуты:
            MODEL_TYPE (str): Тип модели сегментации.
            CHECKPOINT_PATH (str): Путь к файлу весов модели.
        """
        self.MODEL_TYPE = "vit_h"
        self.CHECKPOINT_PATH = 'sam_vit_h_4b8939.pth'

    def model(self, frame_rgb, frame_depth) -> str:
        """
        Метод для обработки RGB и глубинных кадров, генерации масок сегментации и анализа данных глубины.

        :param frame_rgb: RGB изображение для обработки.
        :type frame_rgb: np.ndarray
        :param frame_depth: Изображение глубины для анализа.
        :type frame_depth: np.ndarray
        :return: Строка, содержащая объединенные данные X, Y координат и цветовых значений.
        :rtype: str
        """
        segmenter = SamSegmentation(self.MODEL_TYPE, self.CHECKPOINT_PATH)

        # frame_rgb = cv2.imread('frame_200.jpg')
        # frame_depth = cv2.imread('depth_frame_200.jpg')

        masks = segmenter.segment_image(frame_rgb)

        masks_list = [mask['segmentation'].astype(np.uint8) * 255 for mask in masks]  # Переводим маски в формат, пригодный для OpenCV        

        image_with_annotations = DepthTools.draw_annotations(frame_rgb, masks)

        # Сторим и выводим карту препятсвий
        avg_depth = DepthTools.calculate_average_depth(frame_depth, masks_list)
        depth_img, x_plot, y_plot, color = DepthTools.draw_depth_map(frame_rgb, avg_depth, image_width=640, max_depth=6, show=False, save=True)

        # Соединяем изображения и сохраняем
        height, width, _ = image_with_annotations.shape
        depth_img_resized = cv2.resize(depth_img, (width, height))
        combined_img = np.hstack((image_with_annotations, depth_img_resized))
        cv2.imshow('Combined Image', combined_img)
        cv2.waitKey(0)
        # cv2.imwrite('result_with_boost.jpg', combined_img)
        
        # Конвертация массивов в строки
        x_str = np.array_str(x_plot)
        y_str = np.array_str(y_plot)
        color_str = np.array_str(color)

        # Объединение строк с уникальным разделителем
        delimiter = '|||'
        combined_str = delimiter.join([x_str, y_str, color_str])

        return combined_str
        # return print(x_plot), print(y_plot), print(color)

# if __name__ == "__main__":
#     frame_rgb = cv2.imread('frame_200.jpg')
#     frame_depth = cv2.imread('depth_frame_200.jpg')
#     show_result = ShowResult()
#     show_result.model(frame_rgb, frame_depth)
