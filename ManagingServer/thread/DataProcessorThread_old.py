import threading
import queue
import json
import mysql.connector
from datetime import datetime
from custom_classes import *
from logger_config import setup_logger

logger = setup_logger()

visitors = {} # visitor_id -> Visitor 객체

class DataProcessorThread(threading.Thread):
    def __init__(self, camera_id, data_queue):
        super().__init__()
        self.camera_id = camera_id
        self.data_queue = data_queue  # 카메라 스레드에서 전달받는 데이터를 처리
        self._is_running = True


    def run(self):
        """카메라 데이터 처리"""
        while self._is_running:
            try:
                # 큐에서 데이터 가져오기
                data = self.data_queue.get(timeout=1)  # 1초 대기 후 큐 비어 있으면 예외 발생
                parsed_data = json.loads(data)

                if self.camera_id == "Face":
                    self.process_face_cam(parsed_data)
                elif self.camera_id == "Cart":
                    self.process_cart_cam(parsed_data)
                elif self.camera_id == "Fruit":
                    self.process_fruit_cam(parsed_data)

                print(f"데이터 처리 스레드 (카메라 {self.camera_id}): 데이터 처리 -> {parsed_data}")
                self.data_queue.task_done()  # 처리 완료
            except queue.Empty:
                pass  # 큐가 비어 있으면 대기
            except Exception as e:
                print(f"데이터 처리 스레드 (카메라 {self.camera_id}): 처리 중 오류 발생 -> {str(e)}")

    def stop(self):
        """스레드 중지"""
        self._is_running = False

    def process_fruit_cam(self, data):
        """
        {"apple": 3, "banana": 4, "pear": 5}
        """


    def process_cart_cam(self, data):
        # {"cart_cam_num": 1, "fruits": {"apple": 3, "banana": 4}},
        # {"cart_cam_num": 2, "fruits": {"apple": 4, "pear": 1}},
        # {"cart_cam_num": 3, "fruits": null},
        # {"cart_cam_num": 4, "fruits": null}
        for visit_id in visitors.keys():
            result = [item for item in data if item["cart_cam_num"] == visitors[visit_id].member_id]
            if result:
                visitors[visit_id].cart.update(result[0]["fruits"])
                logger.info(
                    "Member ID: %s, Visit ID: %s, Cart Data: %s",
                    visitors[visit_id].member_id,
                    visit_id,
                    visitors[visit_id].cart.get_data()
                )

    def process_face_cam(self, data):
        member_id = data["member_id"]

        # # MySQL 연결
        # connection = mysql.connector.connect(
        #     host="localhost",         # MySQL 서버 주소 (로컬호스트 또는 IP)
        #     user="root",     # MySQL 사용자 이름
        #     password="your_password", # 비밀번호
        #     database="your_database"  # 사용할 데이터베이스 이름
        # )

        # # 디비에 새 visit_info 생성
        # cursor = connection.cursor()
        # query = "insert into visit_info (member_id, in_dttm) values (%s, %s)"
        # data = (member_id, datetime.now())
        # cursor.execute(query, data)

        # # 새로 만든 visit_info의 visit_id 가져오기
        # cursor.execute("select max(visit_id) from visit_info where member_id=%s", member_id)
        # results = cursor.fetchall()
        # visit_id = results[0][0]

        # # 이번 visit에 대한 cart 만들기
        # cursor.execute("insert into cart (visit_id) values (%s)",  visit_id)
        
        # # cart_id 가져오기
        # cursor.execute("select max(cart_id) from cart where visit_id=%s", visit_id)
        # results = cursor.fetchall()
        # cart_id = results[0][0]

        # # 이번 visit에 대한 event 만들기
        # cursor.execute("insert into event_info (visit_id) values (%s)", visit_id)

        # # event_id 가져오기
        # cursor.execute("select max(event_id) from event where visit_id=%s", visit_id)
        # results = cursor.fetchall()
        # event_id = results[0][0]
        
        # 연결 종료
        # cursor.close()
        # connection.close()

        visit_id = 1
        cart_id = 1

        # 새 cart obj 만들기
        cart = Cart(cart_id)

        # 새 visit obj 만들기
        visitor = Visitor(visit_id, member_id, cart)
        visitors[visit_id] = visitor
        logger.info(f"member_id: {visitor.member_id}, visit_id: {visitor.visit_id}, cart_id: {visitor.cart.cart_id}")






