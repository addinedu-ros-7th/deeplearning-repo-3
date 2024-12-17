import socket
import json
import numpy as np
import time
import datetime
import cv2
import base64

from PyQt5.QtCore import QThread, pyqtSignal

from commons.logger import logger
from commons.socket_setups import server

logger = logger()

class ServerThread(QThread):
    update = pyqtSignal(np.ndarray)
    signin_signal = pyqtSignal(dict)

    def __init__(self, parent=None):
        super(ServerThread, self).__init__(parent)
        self.server_port = server["SERVER_PORT"]
        self.client = server["CLIENT"]
        self.running = True
        # Socket open
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.client, self.server_port))
            logger.error("Server socket has been connected")

            self.server_socket.listen(5)
            logger.info("Server socket is listening")

        except Exception as e :
            logger.error(f"Server socket setup failed : {e}")
            self.server_socket.close()


    def run(self):
        self.client_socket, self.addr = self.server_socket.accept()

        while self.running:
            #logger.info("Waiting data from client")
            try: 
                header = self.client_socket.recv(64).decode('utf-8').strip()
                if not header:
                    logger.info("Header is none")
                    break
                data_type, data_len = header.split('|')
                data_len = int(data_len)
                #logger.info(f"Header : {header}")

                received_data = b""
                while len(received_data) < data_len:
                    packet = self.client_socket.recv(data_len - len(received_data))
                    if not packet:
                        break
                    received_data += packet
                #print(f"data type : {type(received_data)}")
                #print(received_data)

                if data_type == "img":
                    string_data = base64.b64decode(received_data)
                    np_data = np.frombuffer(string_data, np.uint8)
                    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                    self.update.emit(img)
                    cv2.waitKey(1)

                elif data_type == "json":
                    json_data = received_data.decode('utf-8')
                    data = json.loads(json_data)
                    logger.info(f"Received json data: {data}")
                    self.signin_signal.emit(data)

            except Exception as e:
                logger.error(f"Error in receiving data : {e}")
                continue


    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        logger.info("ServerThread has been stopped")
