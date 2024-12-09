import threading
import queue
import json
# import mysql.connector
from datetime import datetime
from custom_classes import *
from logger_config import setup_logger

logger = setup_logger()

class DataProcessorThread(threading.Thread):
    def __init__(self, camera_id, data_queue, visitors, lock, thread_manager):
        super().__init__()
        self.camera_id = camera_id
        self.data_queue = data_queue
        self.visitors = visitors
        self.lock = lock
        self.thread_manager = thread_manager
        self._is_running = True

    def run(self):
        while self._is_running:
            try:
                data = self.data_queue.get(timeout=1)
                parsed_data = json.loads(data)

                if self.camera_id == "Face":
                    self.process_face_cam(parsed_data)
                elif self.camera_id == "Cart":
                    self.process_cart_cam(parsed_data)

                self.data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"처리 중 오류: {str(e)}")


    def process_cart_cam(self, data):
        using_carts = self.thread_manager.get_using_carts()
        logger.info(f"사용 중인 cart_cam: {using_carts}")

        with self.lock:
            for cart_cam, fruits in data.items():
                cart_cam = int(cart_cam)
                if cart_cam in using_carts:  # 사용 중인 카트만 처리
                    # 해당 cart_cam을 사용하는 방문자를 찾음
                    visitor = next(
                        (v for v in self.visitors.values() if v.cart.cart_cam == cart_cam),
                        None,
                    )

                    if visitor:
                        visitor.cart.update(fruits)
                        logger.info(
                            f"Cart Cam {cart_cam} 업데이트: {fruits}, "
                            f"Visitor ID: {visitor.visit_id}"
                        )

    def process_face_cam(self, data):
        member_id = data["member_id"]

        visitor = next(
            (v for v in self.visitors.values() if v.member_id == member_id),
            None,
        )

        if not visitor:
             # get visitor_id, cart_id from database
            visit_id = 1
            cart_id = 1

            cart_cam = self.thread_manager.assign_cart_cam()
            cart = Cart(cart_id=cart_id, cart_cam=cart_cam)
            visitor = Visitor(visit_id, member_id, cart)
            logger.info(f"Visitor 객체 생성: {member_id}, {visit_id}, {cart_id}, {cart_cam}")

            with self.lock:
                self.visitors[visit_id] = visitor

            logger.info(f"Visitor 추가: {member_id} -> Visit ID {visit_id}")
        else:
            # db에 out_dttm이랑 cart정보, purchased 정보 올리고 객체 소멸, 카트 unassign
            self.thread_manager.release_cart_cam(visitor.cart.cart_cam)
            del self.visitors[visitor.visit_id]

            logger.info(f"Visitor 삭제: {member_id} -> Visit ID {visitor.visit_id}")

    def stop(self):
        self._is_running = False
