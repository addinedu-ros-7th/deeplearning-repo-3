import queue
import threading
from ManagingServer.AdminGUI.network.logger_config import setup_logger
from ManagingServer.AdminGUI.network.custom_classes import *

# 로거 가져오기
logger = setup_logger()

carts = {}

class QueueHandler(threading.Thread):
    def __init__(self):
        super().__init__()
        self.data_queue = queue.Queue()  # 작업 큐
        self._is_running = True

    def run(self):
        logger.info("QueueHandler 시작")
        while self._is_running:
            try:
                json_data = self.data_queue.get(timeout=1)  # 1초 타임아웃
                self.handle_data(json_data)
                self.data_queue.task_done()
            except queue.Empty:
                continue  # 큐가 비어있으면 다음 작업으로

    def handle_data(self, json_data):
        try:
            if json_data["camera"] == "FaceCamera":
                # DB에서 members 테이블 확인, visit row 생성, cart row 생성하고 cart_id 가져오기
                cart = Cart(visit_id, cart_id)
                carts[cart_id] = cart

            logger.debug(f"처리 중인 데이터: {json_data}")
        except Exception as e:
            logger.error(f"데이터 처리 중 오류: {str(e)}")

    def enqueue(self, json_data):
        self.data_queue.put(json_data)

    def stop(self):
        self._is_running = False
