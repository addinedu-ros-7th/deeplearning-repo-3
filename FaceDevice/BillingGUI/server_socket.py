import socket

from PyQt5.QtCore import QThread, pyqtSignal

from commons.logger import logger
from commons.socket_setups import server

logger = logger()

class ServerThread(QThread):
    cart_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super(ServerThread, self).__init__(parent)
        self.server_port = server["SERVER_PORT"]
        self.client = server["CLIENT"]

        self.running = True

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self.server_socket:
            try:
                if (self.server_socket.bind((self.client, self.server_port)) == -1):
                    logger.error("Server socket bind() failed")
                    self.server_socket.close()
                    return
            except Exception as e :
                logger.error(f"Server socket bind() failed : {e}")
                self.server_socket.close()
                return
            
            if self.server_socket.listen() == -1:
                logger.error(f"Server socket listen() failed")
                return
            
            self.client_socket, self.addr = self.server_socket.accept()
            with self.client_socket:
                logger.info("Connection to AdminCUI is success")


    def run(self):
        while self.running:
            data = self.client_socket.recv(1024).decode()
            #data = """[{"fruit_name": "Apple", "count": 1, "price": 1000}, {"fruit_name": "Peach", "count": 1, "price": 1000}]"""
            if data is None:
                continue
            self.cart_signal.emit(data)
            logger.info(f"Data has been received from AdminGUI: {data}")
            #break
        self.exec_()


    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        logger.info("ServerThread has been stopped")
