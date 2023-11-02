import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
from yolo.model import YOLOObjectDetector

mask_colors = {}

class DepthTools:
    def show_anns(anns):
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
        # plt.close()
        # return img
    
    def draw_annotations(image, annotations, alpha=0.35):
        global mask_colors

        if len(annotations) == 0:
            return image

        # Копируем изображение, чтобы не изменять оригинал
        image_with_annotations = image.copy()

        sorted_anns = sorted(annotations, key=(lambda x: x['area']), reverse=True)
        # print(sorted_anns)
        # for ann in sorted_anns:
        #     mask = ann['segmentation']
        #     color_mask = np.random.randint(0, 256, 3)  # Создаем случайный цвет
        #     image_with_annotations[mask] = (alpha * color_mask + (1 - alpha) * image_with_annotations[mask]).astype(np.uint8)
        # return image_with_annotations
        for i, ann in enumerate(sorted_anns):
            mask_id = i  # Используем индекс маски в списке как уникальный идентификатор
            if mask_id not in mask_colors:
                mask_colors[mask_id] = np.random.randint(0, 256, 3)  # Создаем и сохраняем случайный цвет для этой маски
            color_mask = mask_colors[mask_id]

            mask = ann['segmentation'].astype(bool)
            image_with_annotations[mask] = (alpha * color_mask + (1 - alpha) * image_with_annotations[mask]).astype(np.uint8)

        return image_with_annotations

    def show_contours(image_rgb, contours):
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

    def calculate_average_depth(depth_input, masks):
        '''
        Считаем среднее значение глубины для каждого объета на карте глубины по маске

        Аргументы:
            depth_input (str/np.ndarray): Путь к фрейму карты глубины или массив numpy с изображением
            masks (list): Маски всех объектов на фрейме
        '''

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
    
    def draw_depth_map(frame, result, image_width, max_depth, show=True, save=False):
        """
        Рисуем карту препятсвий на основе данных о глубине и координатах объектов.

        Аргументы:
            result: Список словарей, каждый из которых содержит данные об объекте
            image_width: Ширина исходного изображения
            max_depth: Максимальное значение глубины для нормализации
        """
        detector = YOLOObjectDetector('yolo/yolo_w/yolov3.cfg', 'yolo/yolo_w/yolov3.weights')
        detections = detector.detect(frame)
        plt.figure(figsize=(10, 5))
        
        # for obj in result:
        #     if obj['left_most_point'] is not None and obj['right_most_point'] is not None:
        #         x = [obj['left_most_point'], obj['right_most_point']]
        #         # y = [obj['average_depth'], obj['average_depth']]
        #         y = [5 - (255 - obj['average_depth']) * 4 / 255, 5 - (255 - obj['average_depth']) * 4 / 255]
        #         plt.plot(x, y, linewidth=5)

        # if show == True:
        #     plt.xlim(0, image_width)
        #     plt.ylim(0, max_depth)
        #     # plt.gca().invert_yaxis()  # Инвертирование оси Y, чтобы глубина увеличивалась вниз
        #     plt.xlabel('Позиция по оси X')
        #     plt.ylabel('Глубина')
        #     plt.title('Карта глубины')
        #     plt.grid(True)
        #     plt.show()
        #     plt.close()

        # if save == True:
        #     buf = io.BytesIO()
        #     plt.savefig(buf, format='png')
        #     buf.seek(0)
        #     img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        #     buf.close()
        #     vis_img = cv2.imdecode(img_arr, 1)
        #     plt.close()
        #     return vis_img
        for obj in result:
            if obj['left_most_point'] is not None and obj['right_most_point'] is not None:
                x = [obj['left_most_point'], obj['right_most_point']]
                y = [5 - (255 - obj['average_depth']) * 4 / 255, 5 - (255 - obj['average_depth']) * 4 / 255]

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
            return vis_img