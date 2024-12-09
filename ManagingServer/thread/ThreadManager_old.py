import queue
from CameraThread import CameraThread
from DataProcessorThread import DataProcessorThread

class ThreadManager:
    def __init__(self):
        super().__init__()
        self.camera_threads = {}
        self.data_processor_threads = {}
        self.data_queues = {}

    def add_camera(self, camera_id, client, port):
        """카메라 스레드 추가 및 시작"""
        if camera_id in self.camera_threads:
            print(f"카메라 {camera_id}는 이미 추가되어 있습니다.")
            return
        
        data_queue = queue.Queue()
        self.data_queues[camera_id] = data_queue

        camera_thread = CameraThread(camera_id, client, port, data_queue) # 카메라의 ip, 포트번호
        camera_thread.start()
        self.camera_threads[camera_id] = camera_thread
        print(f"카메라 {camera_id}: 스레드 시작")

        data_processor_thread = DataProcessorThread(camera_id, data_queue)
        data_processor_thread.start()
        self.data_processor_threads[camera_id] = data_processor_thread

    def remove_camera(self, camera_id):
        """카메라 스레드 중지 및 제거"""
        if camera_id not in self.camera_threads:
            print(f"카메라 {camera_id}는 존재하지 않습니다.")
            return

        camera_thread = self.camera_threads.pop(camera_id)
        camera_thread.stop()
        camera_thread.join()

        data_processor_thread = self.data_processor_threads.pop(camera_id)
        data_processor_thread.stop()
        data_processor_thread.join()

        del self.data_queues[camera_id]

        print(f"카메라 {camera_id}: 카메라 및 데이터 처리 스레드 종료")

    def stop_all(self):
        """모든 카메라 스레드 중지"""
        for camera_id in list(self.camera_threads.keys()):
            self.remove_camera(camera_id)
