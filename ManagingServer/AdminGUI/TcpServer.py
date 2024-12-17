from PyQt5.QtCore import pyqtSignal, QThread, QMetaObject, Qt
from PyQt5.QtNetwork import QTcpServer, QTcpSocket, QAbstractSocket
import json

class TcpServer(QTcpServer):
    def __init__(self, host, port, camera_id, dataProcessor, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self.camera_id = camera_id
        self.dataProcessor = dataProcessor[self.camera_id]
        self.client_threads = []

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
        client_thread = DataRecvThread(self.camera_id, socket_descriptor)

        client_thread.finished.connect(lambda: self.client_threads.remove(client_thread))
        client_thread.finished.connect(client_thread.deleteLater)
        client_thread.dataRecv.connect(self.dataProcessor)

        self.client_threads.append(client_thread)
        print(f"{self.camera_id}'s client num: {len(self.client_threads)}")
        client_thread.start()
        
    def stopServer(self):
        for thread in self.client_threads:
            thread.stop()
            thread.wait()   # 스레드 종료 대기
        self.client_threads.clear()
        super().close()
        print(f"{self.camera_id} Server stopped.")


class DataRecvThread(QThread):
    dataRecv = pyqtSignal(dict)

    def __init__(self, camera_id, socket_descriptor, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.socket_descriptor = socket_descriptor
        self.client_socket = None

    def run(self):
        print(f"{self.camera_id}'s DataRecvThread started...")
        self.client_socket = QTcpSocket()

        if not self.client_socket.setSocketDescriptor(self.socket_descriptor):
            print(f"Error occured while setSocketDescriptor in {self.camera_id} Server")
            return # 스레드 실행 종료. finished 시그널 emit -> 스레드 객체 메모리에서 해제
        
        print(f"New client connected: {self.client_socket.peerAddress().toString()}:{self.client_socket.peerPort()}")

        self.client_socket.readyRead.connect(self.readData)
        self.client_socket.disconnected.connect(self.clientDisconnected)
        self.client_socket.errorOccurred.connect(lambda error: self.clientError(error))

        self.exec_()
           

    def readData(self):
        if not self.client_socket or self.client_socket.state() != QTcpSocket.ConnectedState:
            print("Socket is invalid or disconnected.")
            return
        while self.client_socket.bytesAvailable():
            # PySide2.QtCore.QIODevice.readAll():
            #   Reads all remaining data from the device, and returns it as a byte array.
            print("bytes available")
            data = self.client_socket.readAll().data().decode('utf-8')
            print(f"raw data {data} received")
            json_data = json.loads(data)
            print(f"JSON parsed: {json_data}")
            self.dataRecv.emit(json_data)

    def stop(self):
        self.quit()
        print(f"Quit Thread's event loop")
        if self.client_socket:
            self.client_socket.close()
            self.client_socket.disconnectFromHost()
            self.client_socket.deleteLater() # Schedules this object for deletion.
            self.client_socket = None
        
        print(f"{self.camera_id}'s DataRecvThread stopped.")

    def clientDisconnected(self):
        self.quit()
        print(f"Quit Thread's event loop")
        if self.client_socket:
            self.client_socket.close()
            self.client_socket.deleteLater()
            print(f"{self.camera_id}'s client disconnected. Close Client.")
        
        print(f"Quit Thread")

    def clientError(self, error):
        self.quit()
        print(f"Quit Thread's event loop")
        if error == QAbstractSocket.RemoteHostClosedError:
            print("Error ignored: RemoteHostClosedError already handled")
            return
        #self.client_socket.close()
        if self.client_socket:
            self.client_socket.readyRead.disconnect()
            self.client_socket.close()
            self.client_socket.deleteLater()
            print(f"{self.camera_id}'s client error: {error}. Close Client.")
        

       
