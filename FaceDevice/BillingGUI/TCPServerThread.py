import threading
import queue
import json
import socket

from SignInProcess import CameraThread
from CartProcess import CartThread
from commons.logger import logger

loger = logger()

class TCPServerThread(threading.Thread):
    def __init__(self, client, server_port, data_queue):
        super().__init__()
        self.client = client
        self.server_port = server_port
        self.data_queue = data_queue
        self._is_running = True
        self.socket = None

        
    def run(self):
        """카메라와의 TCP 통신"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
            self.server_socket.bind((self.client, self.server_port))
            self.server_socket.listen() # 서버가 연결을 수락한 상태
            logger.info("TCP 서버 실행 중...")

            while self._is_running:
                try:
                    # 데이터 수신
                    client_socket, addr = self.server_socket.accept()   # 연결 받아들임. (데이터를 보내고 받을 수 있는 새로운 소켓 객체, 연결의 다른 끝에 있는 소켓에 바인드 된 주소)
                    logger.info(f"클라이언트 연결 -> {addr}")

                    try:
                        while True:
                            data = client_socket.recv(1024).decode()
                            if not data:
                                logger.info(f"클라이언트 연결 종료 -> {addr}")
                                break
                            logger.info(f"Receive data from client -> {data}")
                            
                            if data:
                                self.data_queue.put((1, data))
                                logger.info(f"Current data_queue : {self.camera_id}")

                    except Exception as e :
                        break
                except :
                    break
        except Exception as e:
            logger.info(f"Error in starting server: {e}")
        finally:
            self.cleanup()