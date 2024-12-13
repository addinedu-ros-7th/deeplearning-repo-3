import socket
import json
import time

from logger_config import setup_logger

logger = setup_logger()

try:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    server_socket.bind(("0.0.0.0", 6000))
    server_socket.listen() # 서버가 연결을 수락한 상태
    logger.info("TCP Purchase Test 서버 실행 중...")

    try:
        # 데이터 수신
        client_socket, addr = server_socket.accept()   # 연결 받아들임. (데이터를 보내고 받을 수 있는 새로운 소켓 객체, 연결의 다른 끝에 있는 소켓에 바인드 된 주소)
        logger.info(f"클라이언트 연결 -> {addr}")

        try:
            while True:
                data = client_socket.recv(1024).decode()
                if not data:
                    logger.info(f"클라이언트 연결 종료 -> {addr}")
                    break

                logger.info(f"데이터 수신 -> {data}")

        except Exception as e:
            logger.error(f"클라이언트 통신 중 오류 발생 -> {str(e)}")
        finally:
            client_socket.close()
    except Exception as e:
        logger.error(f"클라이언트 연결 처리 중 오류 발생: {str(e)}")
except Exception as e:
    logger.critical(f"서버 시작 중 오류 발생: {str(e)}")
finally:
    """초기화 중 오류, 클라이언트 통신 중 오류, 정상 종료 시"""
    if socket:
        socket.close()
        logger.info(f"서버 소켓 닫기 완료")
        logger.info(f"스레드 종료 및 리소스 정리 완료")
