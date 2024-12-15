import socket
import json

from PyQt5.QtCore import QThread, pyqtSignal

from commons.logger import logger
from commons.socket_setups import client

logger = logger()

class ClientThread(QThread):
    def __init__(self, camera_thread, parent=None):
        super(ClientThread, self).__init__(parent)
        self.server_host = client["SERVER_HOST"]
        self.server_port = client["SERVER_PORT"]

        self.camera_thread  = camera_thread
        self.running = True

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self.client_socket:
            try:
                if (self.client_socket.connect((self.server_host, self.server_port)) == -1):
                    logger.error("Client Socket connect() failed")
                    self.client_socket.close()
                    return
            except Exception as e :
                logger.error(f"Client socket connect() failed : {e}")
                return


    def run(self):
        if self.running:
            logger.info("ClientThread is starting")
            logger.info("Connection to AdminCUI is success")
            self.camera_thread.start()
            self.exec_()
    

    def send(self, data, status):
        if status:
            dict_data = {"camera_id": "Face", "data": [{"member_id": data, "action": "visit"}]}
        else:
            dict_data = {"camera_id": "Purchase", "data": [{"member_id": data, "action": "yes"}]}

        send_data = json.dumps(dict_data).encode()
        self.client_socket.send(send_data)
        logger.info(f"Data has been send to AdminGUI : {dict_data}")


    def stop(self):
        self.running = False
        if self.camera_thread.isRunning():
            self.camera_thread.terminate()
            self.camera_thread.wait()
        self.quit()
        self.wait()
        logger.info("ClientThread has been stopped")
