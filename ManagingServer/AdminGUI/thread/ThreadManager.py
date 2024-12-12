import queue
from .CameraThread import CameraThread
from .DataProcessorThread import DataProcessorThread
import threading
import mysql.connector
from .custom_classes import *
from .logger_config import setup_logger

logger = setup_logger()

class ThreadManager:
    def __init__(self):
        # 3대의 카메라 통신 thread
        self.camera_threads = {}
        # print(DataProcessorThread.__module__)


        # 카메라 통신으로 받은 데이터 저장하는 큐
        self.data_queue = queue.PriorityQueue()
        self.res_data_queue = queue.Queue()

        # Visitor 객체들 저장하는 dict. {visit_id: Visitor}
        self.visitors = {}
        self.lock = threading.Lock()

        # cart 일단 1-4번 사용 가능으로 초기화
        self.available_carts = set(range(1,5))
        self.using_carts = set()

        # DB에 있는 과일 현황 업데이트
        self.fruits = {}
        conn = self.connect_f2mbase()
        cursor = conn.cursor()
        cursor.execute("select fruit_id, stock from fruit")
        for fruit_id, stock in cursor.fetchall():
            self.fruits[fruit_id] = stock

        cursor.execute("""
            SELECT vi.member_id, vi.visit_id, c.cart_id, c.cart_cam, cf.fruit_id, cf.quantity
            FROM visit_info vi
            LEFT JOIN cart c ON vi.visit_id = c.visit_id
            LEFT JOIN cart_fruit cf ON c.cart_id = cf.cart_id
            WHERE vi.out_dttm IS NULL
        """)

        # DB정보에 따라서 현재 매장 내 Visitor와 연결된 cart 업데이트
        data = cursor.fetchall()
        for row in data:
            member_id, visit_id, cart_id, cart_cam, fruit_id, quantity = row
            if visit_id not in self.visitors:
                if cart_id and cart_cam:
                    c = Cart(cart_id, cart_cam)
                    self.using_carts.add(cart_cam)
                else:
                    Cart(None, None)
                v = Visitor(visit_id, member_id, c)
                self.visitors[visit_id] = v
            if fruit_id and quantity:
                self.visitors[visit_id].cart.data[fruit_id] = quantity

        self.available_carts -= self.using_carts

        for v in self.visitors.values():
            logger.info(v)

    
    def connect_f2mbase(self):
        conn = mysql.connector.connect(
            host = "localhost",
            user = "root",
            password = "whdgh29k05",
            database="FruitShopDB"
        )
        return conn
    
    def add_dataprocessor(self):
        data_processor_thread = DataProcessorThread(self.data_queue, self.res_data_queue, self)
        data_processor_thread.start()

    def add_datasender(self, dest_ip, dest_port):
        data_send_thread = DataSendThread(dest_ip, dest_port, self.res_data_queue)
        data_send_thread.start()

    def add_camera(self, camera_id, client, port):
        """카메라 스레드 추가 및 시작"""
        if camera_id in self.camera_threads:
            print(f"카메라 {camera_id}는 이미 추가되어 있습니다.")
            return

        # data_queue = queue.Queue()
        # self.data_queues[camera_id] = data_queue


        # CameraThread 생성
        camera_thread = CameraThread(camera_id, client, port, self.data_queue)
        camera_thread.start()
        self.camera_threads[camera_id] = camera_thread

        print(f"카메라 {camera_id}: 스레드 시작")

    def remove_camera(self, camera_id):
        """카메라 스레드 중지 및 제거"""
        if camera_id not in self.camera_threads:
            print(f"카메라 {camera_id}는 존재하지 않습니다.")
            return

        # 스레드 중지 및 정리
        self._stop_thread(self.camera_threads.pop(camera_id))
        self._stop_thread(self.data_processor_threads.pop(camera_id))

        del self.data_queues[camera_id]

        print(f"카메라 {camera_id}: 종료 완료")

    def assign_cart_cam(self):
        """사용 가능한 카트 할당"""
        with self.lock:
            if not self.available_carts:
                return None  # 사용 가능한 카트가 없음
            cart_cam = self.available_carts.pop()
            self.using_carts.add(cart_cam)
            return cart_cam

    def release_cart_cam(self, cart_cam):
        """사용 중인 카트를 반환"""
        with self.lock:
            if cart_cam in self.using_carts:
                self.using_carts.remove(cart_cam)
                self.available_carts.add(cart_cam)

    def get_using_carts(self):
        """현재 사용 중인 카트 반환"""
        with self.lock:
            return list(self.using_carts)
        
    def convert_keys_to_int(self, d):
        """딕셔너리의 모든 키를 int로 변환"""
        if isinstance(d, dict):
            return {int(k): self.convert_keys_to_int(v) for k, v in d.items()}
        return d  # 값이 딕셔너리가 아니면 그대로 반환

    def stop_all(self):
        """모든 스레드 중지"""
        for camera_id in list(self.camera_threads.keys()):
            self.remove_camera(camera_id)

    def _stop_thread(self, thread):
        """스레드 중지 및 리소스 정리"""
        thread.stop()
        thread.join()
