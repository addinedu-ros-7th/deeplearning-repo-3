import socket
import json
import time

server_ip = '127.0.0.1'  # 서버 IP
server_port = 5002       # 서버 포트

try:
    # 서버에 연결
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))
    print("서버에 연결되었습니다.")


    try:
        # JSON 데이터 생성
        data = [
            {"cart_cam_num": 1, "fruits": {"apple": 3, "banana": 4}},
            {"cart_cam_num": 2, "fruits": {"apple": 4, "pear": 1}},
            {"cart_cam_num": 3, "fruits": None},
            {"cart_cam_num": 4, "fruits": None}
        ]
        # 데이터 송신
        client_socket.send(json.dumps(data).encode())
        print(f"데이터 송신: {data}")
        time.sleep(1)  # 10초 대기
    except (BrokenPipeError, socket.error) as e:
        print(f"송신 오류 발생: {e}. 연결 종료.")

except (socket.error, ConnectionRefusedError) as e:
    print(f"연결 오류: {e}. 서버에 재연결 시도 중...")
    time.sleep(5)  # 재시도 대기

finally:
    client_socket.close()
    print("클라이언트 소켓 닫음.")