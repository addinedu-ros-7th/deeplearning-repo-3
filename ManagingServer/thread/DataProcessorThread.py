import threading
import queue
import json
import mysql.connector
from datetime import datetime
from custom_classes import *
from logger_config import setup_logger

logger = setup_logger()

class DataProcessorThread(threading.Thread):
    def __init__(self, camera_id, data_queue, visitors, lock):
        super().__init__()
        self.camera_id = camera_id
        self.data_queue = data_queue
        self.visitors = visitors
        self.lock = lock
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
        with self.lock:
            for visit_id in self.visitors.keys():
                result = [
                    item for item in data if item["cart_cam_num"] == self.visitors[visit_id].member_id
                ]
                if result:
                    self.visitors[visit_id].cart.update(result[0]["fruits"])
                    logger.info(
                        "Member ID: %s, Visit ID: %s, Cart Data: %s",
                        self.visitors[visit_id].member_id,
                        visit_id,
                        self.visitors[visit_id].cart.get_data(),
                    )

    def process_face_cam(self, data):
        member_id = data["member_id"]
        visit_id = 1
        cart = Cart(cart_id=1)
        visitor = Visitor(visit_id, member_id, cart)

        with self.lock:
            self.visitors[visit_id] = visitor

        logger.info(f"Visitor 추가: {member_id} -> Visit ID {visit_id}")

    def stop(self):
        self._is_running = False
