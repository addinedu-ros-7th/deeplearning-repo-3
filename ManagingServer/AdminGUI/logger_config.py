import logging

# 로거 설정 함수
def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("server.log"),  # 로그를 파일로 저장
            logging.StreamHandler()  # 터미널에 출력
        ]
    )
    return logging.getLogger("TcpServerLogger")
