import socket
import json
import threading
from logger_config import setup_logger

# 로거 가져오기
logger = setup_logger()

class ClientHandlerThread(threading.Thread):
    def __init__(self, client_socket, addr):
        super().__init__()
        self.client_socket = client_socket
        self.addr = addr
        self._is_running = True

    def run(self):
        logger.info(f"클라이언트 스레드 시작: {self.addr}")
        try:
            while self._is_running:
                data = self.client_socket.recv(1024).decode()
                if data:
                    json_data = json.loads(data)
                    logger.debug(f"클라이언트 {self.addr}에서 수신된 데이터: {json_data}")
                else:
                    logger.info(f"클라이언트 {self.addr} 연결 종료")
                    break
        except Exception as e:
            logger.error(f"클라이언트 {self.addr} 처리 중 오류 발생: {str(e)}")
        finally:
            self.client_socket.close()
            logger.info(f"클라이언트 스레드 종료: {self.addr}")

    def stop(self):
        self._is_running = False
        self.client_socket.close()


class TcpServerThread(threading.Thread):
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.server_socket = None
        self._is_running = True
        self.client_threads = []  # 클라이언트 스레드 관리

    def run(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            logger.info("TCP 서버 실행 중...")

            while self._is_running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    logger.info(f"클라이언트 연결: {addr}")

                    # 클라이언트 처리 스레드 생성 및 시작
                    client_thread = ClientHandlerThread(client_socket, addr)
                    client_thread.start()
                    self.client_threads.append(client_thread)

                except Exception as e:
                    logger.error(f"클라이언트 연결 처리 중 오류 발생: {str(e)}")
        except Exception as e:
            logger.critical(f"서버 시작 중 오류 발생: {str(e)}")
        finally:
            self.stop()

    def stop(self):
        self._is_running = False
        if self.server_socket:
            self.server_socket.close()
        # 모든 클라이언트 스레드 종료
        for client_thread in self.client_threads:
            client_thread.stop()
            client_thread.join()
        logger.info("TCP 서버가 중지되었습니다.")
