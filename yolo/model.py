import cv2
import numpy as np

class YOLOObjectDetector:
    """
    Класс YOLOObjectDetector используется для детектирования объектов на изображении с помощью модели YOLO v3 Darknet,
    обученной на наборе данных COCO 2017.

    :param config_path: Путь к файлу конфигурации YOLO.
    :type config_path: str
    :param weights_path: Путь к весам модели YOLO.
    :type weights_path: str
    """
    def __init__(self, config_path, weights_path):
        """
        Инициализация YOLOObjectDetector с заданными конфигурацией и весами модели.
        """
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.classes = ['person', 'bicycle', 'car', 'motorbike', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard','sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'potted plant',
                        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.living = ['person', 'mouse', 'cat', 'dog', 'bird']
        self.nonliving = ['chair', 'table', 'monitor', 'keyboard', 'mouse', 'laptop', 'book', 'bag', 'phone',
                          'bottle', 'cup', 'clock', 'whiteboard', 'projector', 'tv', 'desktop', 'printer', 'poster',
                          'door', 'window', 'notebook', 'pen', 'speaker', 'board', 'professor', 'student']

    def detect(self, frame):
        """
        Метод для детектирования объектов на изображении.

        :param frame: Изображение для детектирования объектов.
        :type frame: np.ndarray
        :return: Список результатов детектирования, включая координаты рамок и метки классов.
        :rtype: list
        """
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        output_layers = self.net.getUnconnectedOutLayersNames()
        layer_outputs = self.net.forward(output_layers)

        boxes = []
        scores = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]
                if confidence > 0.8:
                    cx, cy, w, h = detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    scores.append(confidence)
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.4)
        results = []

        for i in indices:
            box = boxes[i]
            class_id = class_ids[i]
            class_name = self.classes[class_id]
            label = ""
            if class_name in self.living:
                label = 'living'
            elif class_name in self.nonliving:
                label = 'nonliving'

            x, y, w, h = box
            results.append({'box': (x, y, w, h), 'label': label})
            # print(results)
        return results


if __name__ == "__main__":
    detector = YOLOObjectDetector('yolo/yolo_w/yolov3.cfg', 'yolo/yolo_w/yolov3.weights')
    video_path = 'yolo/rgb_output.avi'
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.detect(frame)
        for result in results:
            x, y, w, h = result['box']
            label = result['label']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
