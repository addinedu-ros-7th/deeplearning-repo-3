import socket
import threading
import json
import time
from queue import Empty

from .logger_config import setup_logger

logger = setup_logger()

class DataSendThread(threading.Thread):
    def __init__(self, dest_ip, dest_port, res_data_queue):
        super().__init__()
        self.dest_ip = dest_ip
        self.dest_port = dest_port
        self.res_data_queue = res_data_queue
        self._is_running = True
        self.socket = None

    def run(self):
        """카메라와의 TCP 통신"""
        while self._is_running:
            try:
                # 소켓 연결
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.dest_ip, self.dest_port))
                logger.info(f"{self.dest_ip}에 연결되었습니다.")

                while self._is_running:
                    try:
                        # JSON 데이터 생성 및 송신
                        data = self.res_data_queue.get(timeout=1)
                        # data = {1: 4, 2: 1, 3: 9, 4: 1}
                        logger.info(f"get data out from res_data_queue: {data}")
                        self.socket.send(json.dumps(data).encode())
                        logger.info(f"데이터 송신: {data}")
                    except Empty:
                        # 데이터 큐가 비었을 경우 무시
                        logger.info("Empty res_data_queue")
                        continue
                            # except (BrokenPipeError, socket.error) as e:
                            #     logger.error(f"송신 오류 발생: {e}. 연결 종료 후 재시도.")
                            #     break  # 내부 while 종료 후 재연결 시도

            except (socket.error, ConnectionRefusedError) as e:
                logger.error(f"연결 오류: {e}. 5초 후 재시도...")
                time.sleep(5)  # 재연결 대기

            finally:
                if self.socket:
                    self.socket.close()
                    logger.info("클라이언트 소켓 닫음.")

    def stop(self):
        """스레드 중지"""
        self._is_running = False
        if self.socket:
            self.socket.close()
        logger.info("스레드 중지 요청.")
