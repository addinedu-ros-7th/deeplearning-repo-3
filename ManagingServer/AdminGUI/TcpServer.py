from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtNetwork import QTcpServer, QTcpSocket
import json

class TcpServer(QTcpServer):
    newConnection = pyqtSignal(QTcpSocket)

    def __init__(self, host, port, camera_id, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self.camera_id = camera_id
        self.client_socket = None

    def startServer(self):
        if not self.listen(self.host, self.port):
            print(f"Server failed to start: {self.errorString()}")
        print(f"{self.camera_id} Server started on port {self.port}")

    # PySide2.QtNetwork.QTcpServer.incomingConnection(handle):
    #     called by QTcpServer when a new connection is available.
    #     The base implementation creates a QTcpSocket, 
    #     sets the socket descriptor and then stores the QTcpSocket in an internal list of pending connections. 
    #     Finally newConnection() is emitted.
    def incomingConnection(self, socket_descriptor):
        self.client_socket = QTcpSocket()

        if not self.client_socket.setSocketDescriptor(socket_descriptor):
            print(f"Error occured while setSocketDescriptor in {self.camera_id} Server")
            return  # incomingConnection 종료, 어차피 listen 계속 돌아가서 또 호출됨

        print(f"New client connected: {self.client_socket.peerAddress().toString()}:{self.client_socket.peerPort()}")
        self.newConnection.emit(self.client_socket)
        self.client_socket.errorOccurred.connect(lambda error: self.closeErrorClient(error))
        self.client_socket.disconnected.connect(self.closeClient)
        

    def closeClient(self):
        self.client_socket.close()
        print(f"Client disconnected. Close Client.")

    def closeErrorClient(self, error):
        self.client_socket.close()
        print(f"Client error: {error}. Close Client.")

    def stopServer(self):
        if self.client_socket:
            self.client_socket.disconnectFromHost()
            self.client_socket.waitForDisconnected()

        super().close()
        print(f"{self.camera_id} Server stopped.")


class DataRecvThread(QThread):
    dataRecv = pyqtSignal(QTcpSocket, dict)

    def __init__(self, camera_id, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.client_socket = None
        self.running = False

    def startThread(self, client_socket: QTcpSocket):
        self.client_socket = client_socket
        self.running = True
        self.start()

    def run(self):
        print(f"{self.camera_id}'s DataRecvThread started...")
        while self.running:
            while self.client_socket.bytesAvailable():
                # PySide2.QtCore.QIODevice.readAll():
                #   Reads all remaining data from the device, and returns it as a byte array.
                print("bytes available")
                data = self.client_socket.readAll().data().decode('utf-8')
                print(f"raw data {data} received")
                json_data = json.loads(data)
                print(f"JSON parsed: {json_data}")
                self.dataRecv.emit(self.client_socket, json_data)

    def stop(self):
        self.running = False
        print(f"{self.camera_id}'s DataRecvThread stopped.")
       
