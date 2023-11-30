# from sockets.service_om import ServiceOM
# import time

# def response_handler(response):
#     print(f"Ответ от сервера: {response}")

# if __name__ == '__main__':
#     while True:
#         client = ServiceOM(ip_="localhost", port_=5505)
#         client.run_client(ip="localhost", port=5505, request="Тестовый запрос", response_handler=response_handler)
#         time.sleep(5)

import socket
import numpy as np
import cv2
import base64
import struct


def send_msg(sock: socket, msg: bytes) -> None:
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recvall(sock: socket, n: int) -> bytearray:
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            break
        data.extend(packet)
    return data

def recv_msg(sock: socket) -> bytearray:
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return bytearray()
    msglen = struct.unpack('>I', raw_msglen)[0]
    return recvall(sock, msglen)

server_ip = 'localhost'
server_port = 5505
req = 'get_string'
# self.run_client(ip=server_ip, port=server_port, request=req, response_handler=response_handler)

try:
    while True:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((server_ip, server_port))

        send_msg(client, req.encode("utf-8"))
        response = recv_msg(client)
        response = response.decode("utf-8")

        # print('response', type(response))
        print(response)

        # Разделение строки на составляющие
        x_str, y_str, color_str = response.split('|||')

        x_plot = np.array(eval(x_str))
        y_plot = np.array(eval(y_str))
        color = np.array(eval(color_str))

except Exception as e:
    print(f"Client error when handling client: {e}")
finally:
    client.close()
    print("Connection to server closed")
