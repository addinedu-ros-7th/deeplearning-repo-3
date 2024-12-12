
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

    def run(self):
        logger.info("TCPSender thread started: %s", threading.currentThread().getName())
        while self.running: 
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.server_ip, self.server_port))
                logger.info(f"Connected to {self.server_ip}:{self.server_port}")

                while self.running:
                    try:
                        data = self.data_queue.get(timeout=1)
                        #logger.info("Get data form queue: %s", data)

                        data = {"camera_id": self.camera_id, "member_id": data}
                        self.client_socket.send(json.dumps(data).encode())
                        logger.info(f"Send data : {data}")

                    except queue.Empty:
                        #logger.warning("Queue is empty.")
                        break

            except (BrokenPipeError, socket.error) as e:
                logger.error("Error in TCPSender thread: %s", e)
                time.sleep(5)
            
            finally:
                if self.socket:
                    self.socket.close()
                    logger.info("Client Socket closed")

    def stop(self):
        self.running = False
        logger.info("TCPSender thread stopping")