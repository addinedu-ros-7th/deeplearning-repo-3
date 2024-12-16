import sys
import signal
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtNetwork import QHostAddress

from TcpServer import TcpServer, DataRecvThread
from DataProcessor import DataProcessor


if __name__ == "__main__":
    app = QCoreApplication(sys.argv)

    dataProcessor = DataProcessor()

    # Face Cam Network setting
    faceDataRecvThread = DataRecvThread("Face")
    face_server = TcpServer(host=QHostAddress.Any, port=5001, camera_id="Face")
    face_server.startServer()

    face_server.newConnection.connect(faceDataRecvThread.startThread)
    faceDataRecvThread.dataRecv.connect(dataProcessor.faceProcessor)

    # Fruit Cam Network setting
    fruitDataRecvThread = DataRecvThread("Fruit")
    fruit_server = TcpServer(host=QHostAddress.Any, port=5002, camera_id="Fruit")
    fruit_server.startServer()

    fruit_server.newConnection.connect(fruitDataRecvThread.startThread)
    fruitDataRecvThread.dataRecv.connect(dataProcessor.fruitProcessor)

    # Cart Cam Network setting
    cartDataRecvThread = DataRecvThread("Cart")
    cart_server = TcpServer(host=QHostAddress.Any, port=5003, camera_id="Cart")
    cart_server.startServer()

    cart_server.newConnection.connect(cartDataRecvThread.startThread)
    cartDataRecvThread.dataRecv.connect(dataProcessor.cartProcessor)

    app.aboutToQuit.connect(face_server.stopServer)

    sys.exit(app.exec_())