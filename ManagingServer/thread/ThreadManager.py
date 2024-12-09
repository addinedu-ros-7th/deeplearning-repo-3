import queue
from CameraThread import CameraThread
from DataProcessorThread import DataProcessorThread
import threading

class ThreadManager:
    def __init__(self):
        self.camera_threads = {}
        self.data_processor_threads = {}
        self.data_queues = {}
        self.visitors = {}
        self.fruits = {}
        self.lock = threading.Lock()

        self.available_carts = queue.Queue()
        self.using_carts = queue.Queue()

        for cam_num in range(1,4):
            self.available_carts.put(cam_num)

    def add_camera(self, camera_id, client, port):
        """카메라 스레드 추가 및 시작"""
        if camera_id in self.camera_threads:
            print(f"카메라 {camera_id}는 이미 추가되어 있습니다.")
            return

        data_queue = queue.Queue()
        self.data_queues[camera_id] = data_queue

        # CameraThread 생성
        camera_thread = CameraThread(camera_id, client, port, data_queue)
        camera_thread.start()
        self.camera_threads[camera_id] = camera_thread

        # DataProcessorThread 생성
        data_processor_thread = DataProcessorThread(
            camera_id, data_queue, self.visitors, self.lock, self
        )
        data_processor_thread.start()
        self.data_processor_threads[camera_id] = data_processor_thread

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
        with self.lock:
            if not self.available_carts.empty():
                return self.available_carts.get()
            else:
                return None
            
    def release_cart_cam(self, cam_num):
        with self.lock:
            self.available_carts.put(cam_num)

    def get_using_carts(self):
        with self.lock:
            all_carts = {1,2,3,4}
            available_carts = set(list(self.available_carts.queue))
            return list(all_carts - available_carts)

    def stop_all(self):
        """모든 스레드 중지"""
        for camera_id in list(self.camera_threads.keys()):
            self.remove_camera(camera_id)

    def _stop_thread(self, thread):
        """스레드 중지 및 리소스 정리"""
        thread.stop()
        thread.join()
