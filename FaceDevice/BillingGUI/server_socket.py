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



    def accept_connection(self):
        try:
            self.client_socket, self.addr = self.server_socket.accept()
            logger.info("Waiting data from client")
        except Exception as e:
            logger.info(f"Error accepting connection: {e}")

    def run(self):
        while self.running:
            self.accept_connection()

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
