import sys
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot , QDate
from DBConnector import DBThread
from PoseDetector import VideoThread
from thread.custom_classes import Person,Visitor,Event
from thread.ThreadManager import ThreadManager
import pymysql

#logger = setup_logger()

# UI 파일을 로드하여 from_class로 정의
from_class = uic.loadUiType("AdminGUI.ui")[0]


# 메인 윈도우 클래스
class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Detection")  # 윈도우 제목 설정
        self.setupUi(self)  # UI 초기화

        # 메인 스레드에서 person_states 딕셔너리 초기화
        self.person_states = {}
        
        # 비디오 스레드 시작
        self.thread = VideoThread()
        
        # 스레드에서 전달받은 신호를 슬롯에 연결
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.person_entered.connect(self.person_entered)
        self.thread.person_posed.connect(self.person_posed)
        self.thread.person_stopped_posing.connect(self.person_stopped_posing)
        self.thread.person_exited.connect(self.person_exited)
        self.thread.file_path.connect(self.save_video_path)  # 경로 신호 연결
        self.thread.start()  # 스레드 실행

        self.thread_manager = ThreadManager()

        # DataProcessorThread 시작
        self.thread_manager.add_dataprocessor()

        # 카메라 추가
        self.thread_manager.add_camera(camera_id="Face", client="0.0.0.0", port=5001)
        self.thread_manager.add_camera(camera_id="Cart", client="0.0.0.0", port=5002)
        self.thread_manager.add_camera(camera_id="Fruit", client="0.0.0.0", port=5003)

        self.thread_manager.add_datasender(dest_ip="192.168.0.74", dest_port=5005)
        

        # DB 스레드 초기화
        self.db_thread = DBThread({
            "host": "localhost",
            "user": "root",
            "password": "whdgh29k05",
            "db": "FruitShopDB",
            "charset": "utf8mb4",
        })

        # QDateEdit 신호 연결
        self.dateEdit.dateChanged.connect(self.filter_by_date)

        # 테이블 업데이트 신호 연결
        self.db_thread.selling_log_signal.connect(self.update_selling_log)
        self.db_thread.visit_log_signal.connect(self.update_visit_log)
        self.db_thread.event_log_signal.connect(self.update_event_log)
        self.db_thread.selling_sum_signal.connect(self.update_selling_sum)
        self.db_thread.start()

    #-------------------------------------------------------------------로그 GUI 관련 함수

    def filter_by_date(self):
        """선택된 날짜를 기준으로 데이터 필터링"""
        selected_date = self.dateEdit.date()  # QDate 객체
        formatted_date = selected_date.toString("yyyy-MM-dd")  # MySQL 쿼리에 맞는 형식

        # DB 스레드에 필터 요청
        self.db_thread.set_date_filter(formatted_date)

    def update_selling_log(self, data):
        """판매 로그를 테이블 위젯에 업데이트"""
        self.selling_table.setRowCount(len(data))
        for row_idx, row_data in enumerate(data):
            for col_idx, col_data in enumerate(row_data):
                self.selling_table.setItem(row_idx, col_idx, QTableWidgetItem(str(col_data)))

    def update_visit_log(self, data):
        """방문 로그 업데이트"""
        self.visit_table.setRowCount(len(data))
        #print("visit log is:", data)
        for row_idx, row_data in enumerate(data):
            for col_idx, col_data in enumerate(row_data):
                self.visit_table.setItem(row_idx, col_idx, QTableWidgetItem(str(col_data)))

    def update_event_log(self, data):
        """이벤트 로그 업데이트"""
        self.event_table.setRowCount(len(data))
        for row_idx, row_data in enumerate(data):
            for col_idx, col_data in enumerate(row_data):
                self.event_table.setItem(row_idx, col_idx, QTableWidgetItem(str(col_data)))

    def update_selling_sum (self,data) :
        """매장 총 매출 업데이트"""
        #print(f"Received selling_sum: {data}")
        self.sellingsum.setText(f"{data}원")
        self.sellingsum2.setText(f"{data}원")

    def closeEvent(self, event):
        """윈도우 종료 시 DB 스레드 종료"""
        self.db_thread.stop()
        super().closeEvent(event)

    # OpenCV 이미지를 QPixmap으로 변환하여 라벨 크기에 맞게 조정하는 함수
    def convert_cv_qt(self, cv_img, label_width, label_height):
        """OpenCV 이미지를 QPixmap으로 변환하고 라벨 크기에 맞게 조정"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # 색상 변환
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qt_image = qt_image.scaled(label_width, label_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(qt_image)

    #----------------------------------------------------------------------- 사람포즈인식 관련 함수

    # OpenCV 이미지를 QLabel에 업데이트
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """OpenCV 이미지를 QLabel에 업데이트"""
        label1_width = self.CCTV_1.width()
        label1_height = self.CCTV_1.height()
        label2_width = self.CCTV_2.width()
        label2_height = self.CCTV_2.height()

        qt_img1 = self.convert_cv_qt(cv_img, label1_width, label1_height)
        qt_img2 = self.convert_cv_qt(cv_img, label2_width, label2_height)

        self.CCTV_1.setPixmap(qt_img1)
        self.CCTV_2.setPixmap(qt_img2)

    # 비디오 스레드로부터 사람의 상태 변화를 처리하는 슬롯
    @pyqtSlot(int)
    def person_entered(self, track_id):
        """사람이 들어왔을 때 호출"""
        if track_id not in self.person_states:
            self.person_states[track_id] = Person(track_id)
            self.person_states[track_id].detected = True
            
            try:
                # MySQL 데이터베이스 연결
                conn = pymysql.connect(
                    host="localhost",
                    user="root",
                    password="whdgh29k05",
                    db="FruitShopDB",
                    charset="utf8mb4"
                )
                cursor = conn.cursor()

                # 방문 기록에서 가장 최근 visit_id 가져오기
                query = """
                    SELECT visit_info.visit_id
                    FROM visit_info
                    ORDER BY visit_id DESC
                    LIMIT 1;
                """
                cursor.execute(query)
                result = cursor.fetchone()  # 결과가 한 줄일 경우 fetchone 사용
                if result:
                    visit_id = result[0]
                    self.person_states[track_id].visit_id = visit_id
                    print(f"Person {track_id}, {visit_id} entered")
                else:
                    print("No visit_id found in cart_fruit table")

            except pymysql.MySQLError as e:
                print(f"MySQL Error: {e}")

            finally:
                # 커서 및 연결 닫기
                if 'cursor' in locals():
                    cursor.close()
                if 'conn' in locals() and conn.open:
                    conn.close()

    @pyqtSlot(int)
    def person_posed(self, track_id):
        """사람이 포즈를 취했을 때 호출"""
        if track_id in self.person_states:
            self.person_states[track_id].posed = True
            print(f"Person {track_id} posed")

    @pyqtSlot(int)
    def person_stopped_posing(self, track_id):
        """사람이 포즈를 멈췄을 때 호출"""
        if track_id in self.person_states:
            self.person_states[track_id].posed = False
            print(f"Person {track_id} stopped posing")

    @pyqtSlot(int)
    def person_exited(self, track_id):
        """사람이 나갔을 때 호출"""
        if track_id in self.person_states:
            del self.person_states[track_id]
            print(f"Person {track_id} exited")

    @pyqtSlot(str)
    def save_video_path(self, path):
        """앞으로 저장될 비디오 경로를 업데이트"""
        self.saved_video_path = path
        print(f"앞으로 저장될 비디오 경로: {path}")

    # 윈도우 종료 시 비디오 스레드 중지
    def closeEvent(self, event):
        """윈도우 종료 시 호출 - 비디오 스레드 정지"""
        self.thread.stop()
        self.thread_manager.stop_all()
        if hasattr(self, "tcp_server_thread") and self.tcp_server_thread.is_alive():
            self.tcp_server_thread.stop()  # TCP 서버 종료 플래그 설정
            self.tcp_server_thread.join()  # 스레드가 종료될 때까지 대기

        event.accept()

# 메인 함수
if __name__ == "__main__":
    app = QApplication(sys.argv)  # PyQt 애플리케이션 생성
    myWindow = WindowClass()      # 메인 윈도우 인스턴스 생성
    myWindow.show()               # 윈도우 표시
    sys.exit(app.exec_())         # 애플리케이션 실행
