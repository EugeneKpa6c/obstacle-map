from show import ShowResult

from sockets.service import Service
from sockets.cam import Camera

import cv2
import logging

logging.basicConfig(level=logging.INFO)

class ServiceOM(Service):
    """
    Класс ServiceOM расширяет функциональность базового класса Service,
    предоставляя обработку видеопотока и построение карты препятсвий.
    """

    def _do_job(self) -> str:
        """
        Переопределенный метод, выполняющий основную работу сервиса.

        Включает в себя подключение к видеопотоку, обработку кадров и выполнение специализированных задач.
        В случае исключений, передает их в обработчик запросов.

        :return: Результат обработки или строку с ошибкой.
        :rtype: str
        """
        try:
            # Подключение к RTSP потоку камеры
            url = 'rtsp://0.0.0.0:8554/mystream'  # rtsp-стрим
            url = 0  # webcam
            cap = Camera(url)        

            # Инициализация переменных
            self.__init_vars()

            # "Бесконечный цикл", который выполняют работу, для которой создан сервис:
            # 1) Обработка кадров
            # 2) Обработка звука
            # 3) Обработка сигналов (с лидаров/сонаров)
            while True:
                if self.need_job_break:
                    return
                if not self.need_job_pause:
                    continue
                # Конец

                # Получение кадров из потока
                frame_raw = cap.getFrame()
                # Проверка, что кадр непустой
                if frame_raw is None:
                    continue

                # # На текущий момент фрейм - это две картинки с RGB-сенсора и со стереопары (Depth) w=1280px, h=480px
                # # RGB фрейм слева, т.е. его координаты (x=0, y=0, w=640, h=480)
                # # Depth фрейм cghfdf, т.е. его координаты (x=640, y=0, w=640, h=480)
                h, w, ch = frame_raw.shape
                w = w//2
                self.frame_rgb = frame_raw[0:h, 0:w]  # получение RGB
                self.frame_depth = frame_raw[0:h, w:2*w]  # получение Depth

                # Обработка кадра (или иная работа сервиса)
                # В данном примере работа - это определение лица и его идентификация между кадрами
                result = self.__specific_work(self.frame_rgb, self.frame_depth)
                 
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

                return result

        except Exception as e:
            self._request_handler(e)


        finally:    
            cv2.destroyAllWindows()
            # close job + server
            self.stop()

    def __specific_work(self, frame_rgb, frame_depth) -> str:
        """
        Приватный метод для выполнения специфической работы с кадрами.

        :param frame_rgb: Кадр в RGB формате.
        :type frame_rgb: np.ndarray
        :param frame_depth: Кадр с данными глубины.
        :type frame_depth: np.ndarray
        :return: Результат специфической работы с кадрами.
        :rtype: str
        """
        return ShowResult.model(self, frame_rgb, frame_depth)

    def _request_handler(self, request: str) -> str:
        """
        Переопределенный метод для обработки входящих запросов.

        :param request: Входящий запрос.
        :type request: str
        :return: Ответ на запрос.
        :rtype: str
        """
        if request.lower() == "get_map":
            result_string = self._do_job()
            return result_string

