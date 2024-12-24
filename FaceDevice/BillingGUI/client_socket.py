import socket
import json

from PyQt5.QtCore import QThread, pyqtSignal

from commons.logger import logger
from commons.socket_setups import client

logger = logger()

class ClientThread(QThread):
    cart_signal = pyqtSignal(str)

    def __init__(self, camera_thread, parent=None):
        super(ClientThread, self).__init__(parent)
        self.server_ip = client["SERVER_IP"]
        self.server_port = client["SERVER_PORT"]
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.server_ip, self.server_port))
            logger.error("Client socket has been connected")

        except Exception as e :
            self.client_socket.close()
            logger.error(f"Client socket setup failed : {e}")

        self.camera_thread  = camera_thread
        self.recv_thread = RecvThread(self.cart_signal, self.client_socket)
        self.running = True

    def run(self):
        if self.running:
            logger.info("ClientThread is starting")
            self.camera_thread.start()
            self.recv_thread.start()
            self.exec_()
    
    def send(self, data, status):
        if status == False:
            dict_data = {"camera_id": "Face", "data": [{"member_id": data, "action": "visit"}]}
        else:
            dict_data = {"camera_id": "Purchase", "data": [{"member_id": data, "action": "yes"}]}

        send_data = json.dumps(dict_data).encode()
        self.client_socket.send(send_data)
        logger.info(f"Data has been sent to server : {dict_data}")

    def stop(self):
        self.running = False
        if self.recv_thread.running:
            self.recv_thread.stop()
        self.quit()
        self.wait()
        logger.info("ClientThread has been stopped")


class RecvThread(QThread):
    def __init__(self, cart_signal, client_socket=None, parent=None):
        super(RecvThread, self).__init__(parent)
        self.client_socket = client_socket
        self.cart_signal = cart_signal
        self.running = True

    def run(self):
        logger.info("RecvThread is starting")
        while self.running:
            try:
                data = self.client_socket.recv(1024).decode()
                if data is None:
                    continue
                self.cart_signal.emit(data)
                logger.info(f"Data has been received from server: {data}")
                break
            except Exception as e:
                #logger.info(f"Error receiving data : {e}")
                continue

    def stop(self):
        self.running = False
        logger.info("RecvThread has been stopped")
