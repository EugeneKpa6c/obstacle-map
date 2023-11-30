from sockets.service_om import ServiceOM


if __name__ == '__main__':
    # Для запуска сервиса используем
    # IP-адрес - строковый литерал
    # Порт - int value
    # service_var = ServiceOM(ip_="10.10.10.179", port_=5505)
    service_var = ServiceOM(ip_="localhost", port_=5505)

    # Запуск сервиса
    # Сервис состоит из основной работы 
    # (для чего сервис создан? Построение карты, поиск предметов, классификация жестов, распознавание речи)
    # и из сервера, к которому могут обращаться другие сервисы
    service_var.start()
