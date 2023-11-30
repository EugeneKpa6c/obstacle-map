import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
from yolo.model import YOLOObjectDetector

mask_colors = {}

class DepthTools:
    """
    Класс DepthTools предназначен для работы с изображениями и данными глубины, включая визуализацию и анализ.
    """
    def show_anns(anns: list):
        """
        Метод для отображения аннотаций на изображении.

        :param anns: Список аннотаций для визуализации.
        :type anns: list
        """
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
    
    def draw_annotations(image: np.ndarray, annotations: list, alpha=0.35) -> np.ndarray:
        """
        Метод для отрисовки аннотаций на изображении.

        :param image: Изображение, на котором будут отрисованы аннотации.
        :type image: np.ndarray
        :param annotations: Список аннотаций для отрисовки.
        :type annotations: list
        :param alpha: Прозрачность аннотаций.
        :type alpha: float, optional
        :return: Изображение с наложенными аннотациями.
        :rtype: np.ndarray
        """
        global mask_colors

        if len(annotations) == 0:
            return image

        # Копируем изображение, чтобы не изменять оригинал
        image_with_annotations = image.copy()

        sorted_anns = sorted(annotations, key=(lambda x: x['area']), reverse=True)

        for i, ann in enumerate(sorted_anns):
            mask_id = i  # Используем индекс маски в списке как уникальный идентификатор
            if mask_id not in mask_colors:
                mask_colors[mask_id] = np.random.randint(0, 256, 3)  # Создаем и сохраняем случайный цвет для этой маски
            color_mask = mask_colors[mask_id]

            mask = ann['segmentation'].astype(bool)
            image_with_annotations[mask] = (alpha * color_mask + (1 - alpha) * image_with_annotations[mask]).astype(np.uint8)

        return image_with_annotations

    def show_contours(image_rgb: np.ndarray, contours: list) -> list:
        """
        Метод для отображения контуров на RGB изображении.

        :param image_rgb: RGB изображение для визуализации контуров.
        :type image_rgb: np.ndarray
        :param contours: Список контуров для отображения.
        :type contours: list
        """
        plt.figure(figsize=(20, 20))
        plt.imshow(image_rgb)
        
        for contour in contours:
            # Склеиваем все контуры в один массив для удобства отрисовки
            all_contours = np.concatenate(contour, axis=0)
            
            # Рисуем контур на изображении
            plt.plot(all_contours[:, 0, 0], all_contours[:, 0, 1], linewidth=2)
            
        plt.axis('off')
        plt.show()
        plt.close()

    def calculate_average_depth(depth_input: np.ndarray, masks: list) -> list:
        """
        Метод для расчета средней глубины каждого объекта на карте глубины по маске.

        :param depth_input: Карта глубины или путь к изображению карты глубины.
        :type depth_input: np.ndarray
        :param masks: Маски всех объектов на фрейме.
        :type masks: list
        :return: Список средних глубин и других связанных данных для каждого объекта.
        :rtype: list
        """

        if isinstance(depth_input, str):
            depth_map = cv2.imread(depth_input, cv2.IMREAD_UNCHANGED)
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
            if depth_map is None:
                print("Не удалось загрузить карту глубины")
                return None
        elif isinstance(depth_input, np.ndarray):
            depth_map = depth_input
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
        else:
            print("Неверный тип входных данных. Ожидается строка (путь к файлу) или массив NumPy.")
            return None

        depth_map = depth_map.astype(np.float32)

        result = []
        for mask in masks:
            isolated_region = cv2.bitwise_and(depth_map, depth_map, mask=mask)  
            masked_array = np.ma.masked_equal(isolated_region, 0)
            
            # Рассчитаем среднее значение и стандартное отклонение
            mean = masked_array.mean()
            std = masked_array.std()
            
            # Отфильтруем выбросы
            filtered_array = np.ma.masked_outside(masked_array, mean - 2 * std, mean + 2 * std)
            
            # Рассчитаем среднее значение отфильтрованной области
            average_depth = filtered_array.mean()
            
            # Найдем координаты самой левой и правой точек маски
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x_coords = coords[:,:,0]
                left_most_point = np.min(x_coords)
                right_most_point = np.max(x_coords)
            else:
                left_most_point = None
                right_most_point = None
            
            result.append({
                'average_depth': average_depth,
                'left_most_point': left_most_point,
                'right_most_point': right_most_point
            })

        return result
    
    def draw_depth_map(frame: np.ndarray, result: list, image_width: int, max_depth: int, show=True, save=False):
        """
        Метод для рисования карты препятствий на основе данных о глубине и координатах объектов.

        :param frame: Исходное изображение.
        :type frame: np.ndarray
        :param result: Список словарей с данными об объектах.
        :type result: list
        :param image_width: Ширина исходного изображения.
        :type image_width: int
        :param max_depth: Максимальное значение глубины для нормализации.
        :type max_depth: int
        :param show: Флаг для отображения графика.
        :type show: bool, optional
        :param save: Флаг для сохранения графика.
        :type save: bool, optional
        :return: Визуализация карты препятствий, координаты X и Y, цвета.
        :rtype: tuple
        """
        detector = YOLOObjectDetector('yolo/yolo_w/yolov3.cfg', 'yolo/yolo_w/yolov3.weights')
        detections = detector.detect(frame)
        plt.figure(figsize=(10, 5))

        x_plot, y_plot, clr = [], [], []
        for obj in result:
            if obj['left_most_point'] is not None and obj['right_most_point'] is not None:
                x = [obj['left_most_point'], obj['right_most_point']]
                y = [5 - (255 - obj['average_depth']) * 4 / 255, 5 - (255 - obj['average_depth']) * 4 / 255]
                x_plot.append(x)
                y_plot.append(y)

                # Проверяем, попадает ли 'x' полностью в какой-либо 'box' обнаруженный YOLO
                in_living_box = False
                for detection in detections:
                    if detection['label'] == 'living':
                        box_x1, box_y1, box_w, box_h = detection['box']
                        box_x2 = box_x1 + box_w

                        # Проверяем, что левая и правая точки находятся внутри области бокса
                        if box_x1 <= obj['left_most_point'] and box_x2 >= obj['right_most_point']:
                            in_living_box = True
                            break

                # Выбираем цвет на основе сопоставления с детекцией YOLO
                color = 'red' if in_living_box else 'grey'
                clr.append(color)
                plt.plot(x, y, linewidth=5, color=color)

        if show:
            plt.xlim(0, image_width)
            plt.ylim(0, max_depth)
            plt.xlabel('Позиция по оси X')
            plt.ylabel('Глубина')
            plt.title('Карта глубины')
            plt.grid(True)
            plt.show()
            plt.close()

        if save:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()
            vis_img = cv2.imdecode(img_arr, 1)
            plt.close()
            return vis_img, x_plot, y_plot, clr