import socket
import threading

from logger_config import setup_logger

logger = setup_logger()

class CameraThread(threading.Thread):
    def __init__(self, camera_id, client, server_port, data_queue):
        super().__init__()
        self.camera_id = camera_id
        self.client = client
        self.server_port = server_port
        self.data_queue = data_queue
        self._is_running = True
        self.socket = None

    def run(self):
        """카메라와의 TCP 통신"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
            self.server_socket.bind((self.client, self.server_port))
            self.server_socket.listen()
            logger.info("TCP 서버 실행 중...")

            while self._is_running:
                try:
                    # 데이터 수신
                    client_socket, addr = self.server_socket.accept()
                    logger.info(f"카메라 {self.camera_id}: 클라이언트 연결 -> {addr}")

                    try:
                        while True:
                            data = client_socket.recv(1024).decode()
                            if not data:
                                logger.info(f"카메라 {self.camera_id}: 클라이언트 연결 종료 -> {addr}")
                                break
                            logger.info(f"카메라 {self.camera_id}: 데이터 수신 -> {data}")
                            self.data_queue.put(data)

                    except Exception as e:
                        logger.error(f"카메라 {self.camera_id}: 클라이언트 통신 중 오류 발생 -> {str(e)}")
                    finally:
                        client_socket.close()
                except Exception as e:
                    logger.error(f"클라이언트 연결 처리 중 오류 발생: {str(e)}")
        except Exception as e:
            logger.critical(f"서버 시작 중 오류 발생: {str(e)}")
        finally:
            """초기화 중 오류, 클라이언트 통신 중 오류, 정상 종료 시"""
            self.cleanup()

    def send_data(self, message):
        """데이터 전송"""
        try:
            if self.socket:
                self.socket.sendall(message.encode())
                print(f"카메라 {self.camera_id}: 데이터 전송 -> {message}")
        except Exception as e:
            print(f"카메라 {self.camera_id}: 데이터 전송 중 오류 -> {str(e)}")

    def stop(self):
        """스레드 중지"""
        self._is_running = False
        if self.socket:
            self.socket.close()
        logger.info("TCP 서버가 중지되었습니다.")

    def cleanup(self):
        """리소스 정리"""
        if self.socket:
            self.socket.close()
            logger.info(f"카메라 {self.camera_id}: 서버 소켓 닫기 완료")
        logger.info(f"카메라 {self.camera_id}: 스레드 종료 및 리소스 정리 완료")