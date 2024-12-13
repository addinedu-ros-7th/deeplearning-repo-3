
import threading
import socket
import json
import time
import queue
from commons.logger import logger

logger = logger()


#-------------------Variable Setting-------------------
SERVER_IP = '192.168.0.100'
SERVER_PORT = 5001
CAMERA_ID = "Face"
# -----------------------------------------------------

class TCPSenderThread(threading.Thread):
    def __init__(self, 
                 data_queue,
                 server_ip=SERVER_IP, 
                 server_port=SERVER_PORT, 
                 camera_id=CAMERA_ID):
        super().__init__()
        self.server_ip = server_ip
        self.server_port = server_port

        self.camera_id = camera_id
        self.data_queue = data_queue
        self.running = True
        self.socket = None
        logger.info("TCPSenderThread starting: %s", threading.currentThread().getName())

    def run(self):
        logger.info("TCPSenderThread running: %s", threading.currentThread().getName())
        while self.running: 
            if not self.data_queue.empty():
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.server_ip, self.server_port))
                #logger.info(f"Connected to {self.server_ip}:{self.server_port}")

                data = self.data_queue.get(timeout=1)
                logger.info("Get data from queue: %s", data)
                dict_data = {"camera_id": self.camera_id, "data":[{"member_id": data, "action":"visit"}]}

                self.client_socket.send(json.dumps(dict_data).encode())
                logger.info(f"Send data : {dict_data}")

            else:
                #logger.info("Waiting Face recognition")
                time.sleep(1)
                continue
            
    def stop(self):
        self.running = False
        logger.info("TCPSenderThread stopping")