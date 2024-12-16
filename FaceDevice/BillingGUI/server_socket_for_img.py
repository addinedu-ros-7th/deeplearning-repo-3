import socket
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
    signin_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super(ServerThread, self).__init__(parent)
        self.server_port = server["SERVER_PORT"]
        self.client = server["CLIENT"]
        # Socket open
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.client, self.server_port))
            logger.error("Server socket has been connected")

            self.server_socket.listen(5)
            logger.info("Server socket is listening")

            self.client_socket, self.addr = self.server_socket.accept()

        except Exception as e :
            logger.error(f"Server socket setup failed : {e}")
            self.server_socket.close()

        self.running = True


    def run(self):
        logger.info("ServerThread is starting")
        while self.running:
            try:
                length = self.recvall(self.client_socket, 64)
                length1 = length.decode('utf-8')
                stringData = self.recvall(self.client_socket, int(length1))
                data = np.frombuffer(base64.b64decode(stringData), np.uint8)
                decimg = cv2.imdecode(data, 1)

                #print(f"Received img : {type(decimg)}")
                self.update.emit(decimg)
                #cv2.imshow("image", decimg)
                cv2.waitKey(1)
                
            except Exception as e:
                print(f"Error in receiving images : {e}")
                continue
                

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        logger.info("ServerThread has been stopped")
