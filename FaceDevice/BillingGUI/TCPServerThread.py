import threading
import queue
import json
import socket
import time

from SignInProcess import CameraThread
from CartProcess import CartThread
from commons.logger import logger

logger = logger()
SERVER_PORT = 5005
CLIENT = "0.0.0.0"

class TCPServerThread(threading.Thread):
    def __init__(self, data_queue, client=CLIENT, server_port=SERVER_PORT):
        super().__init__()
        self.client = client
        self.server_port = server_port
        self.data_queue = data_queue
        self.running = True
        self.socket = None
        logger.info(f"TCPServerThread starting: {threading.currentThread().getName()}")
        
    def run(self):
        logger.info(f"TCPServerThread running: {threading.currentThread().getName()}")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
        self.server_socket.bind((self.client, self.server_port))
        self.server_socket.listen() # 서버가 연결을 수락한 상태
        
        #while self.running:
        client_socket, addr = self.server_socket.accept()   # 연결 받아들임. (데이터를 보내고 받을 수 있는 새로운 소켓 객체, 연결의 다른 끝에 있는 소켓에 바인드 된 주소)
        logger.info(f"서버 연결 -> {addr}")

        while self.running:
            logger.info("Waiting data from server")
            data = client_socket.recv(1024).decode()
            if not data:
                logger.info(f"Data is none -> {addr}")
                continue
            logger.info(f"Receive data from server -> {data}")
            try:
                self.data_queue.put(data)
                print(f"Put data into queue : {len(self.data_queue)}")
            except Exception as e:
                logger.info("Error in data_queue.put()")
                time.sleep(1)
                continue
            #if self.data_queue.empty():
            #    continue


    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()
        logger.info("socket closed")
