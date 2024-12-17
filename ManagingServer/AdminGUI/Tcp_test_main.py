import sys
import signal
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtNetwork import QHostAddress

from TcpServer import TcpServer, DataRecvThread
from DataProcessor import DataProcessor


if __name__ == "__main__":
    app = QCoreApplication(sys.argv)

    dataProcessor = DataProcessor()
    face_server = TcpServer(host=QHostAddress.AnyIPv4, port=5001, camera_id="Face", dataProcessor=dataProcessor.processors)
    face_server.startServer()


    cart_server = TcpServer(host=QHostAddress.AnyIPv4, port=5002, camera_id="Cart", dataProcessor=dataProcessor.processors)
    cart_server.startServer()


    fruit_server = TcpServer(host=QHostAddress.AnyIPv4, port=5003, camera_id="Fruit", dataProcessor=dataProcessor.processors)
    fruit_server.startServer()

    app.aboutToQuit.connect(face_server.stopServer)

    sys.exit(app.exec_())