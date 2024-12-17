import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtNetwork import QHostAddress
from TcpServer import TcpServer
from DataProcessor import DataProcessor



# UI 파일을 로드하여 from_class로 정의
from_class = uic.loadUiType("AdminGUI.ui")[0]


# 메인 윈도우 클래스
class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Detection")  # 윈도우 제목 설정
        self.setupUi(self)  # UI 초기화

        # Camera data processor
        self.dataProcessor = DataProcessor()
        self.face_server = TcpServer(host=QHostAddress.AnyIPv4, port=5001, camera_id="Face", dataProcessor=self.dataProcessor.processors)
        self.face_server.startServer()


        self.cart_server = TcpServer(host=QHostAddress.AnyIPv4, port=5002, camera_id="Cart", dataProcessor=self.dataProcessor.processors)
        self.cart_server.startServer()


        self.fruit_server = TcpServer(host=QHostAddress.AnyIPv4, port=5003, camera_id="Fruit", dataProcessor=self.dataProcessor.processors)
        self.fruit_server.startServer()



# 메인 함수
if __name__ == "__main__":
    app = QApplication(sys.argv)  # PyQt 애플리케이션 생성
    myWindow = WindowClass()      # 메인 윈도우 인스턴스 생성
    myWindow.show()               # 윈도우 표시
    sys.exit(app.exec_())         # 애플리케이션 실행
