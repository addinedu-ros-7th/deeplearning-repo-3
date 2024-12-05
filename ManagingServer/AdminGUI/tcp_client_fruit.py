import socket
import json
import time

server_ip = 'server'  # 서버 IP
server_port = 8080       # 서버 포트

cnt = 0

while cnt < 50:
    try:
        # 서버에 연결
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((server_ip, server_port))
        print("서버에 연결되었습니다.")

        while cnt < 50:
            try:
                # JSON 데이터 생성
                data = {
                    "camera_id": 21,
                    "apple_defective" :3,
                    "apple_fair" :3,
                    "peach_defective" :3,
                    "peach_fair" :3,
                    "pomegranate_defective": 2,
                    "pomegranate_fair" : 3,
                    "mandarin_defective" : 3,
                    "mandarin_fair" : 3
                }
                # 데이터 송신
                client_socket.send(json.dumps(data).encode())
                print(f"데이터 송신: {data}")
                time.sleep(1)  # 10초 대기
                cnt += 1
            except (BrokenPipeError, socket.error) as e:
                print(f"송신 오류 발생: {e}. 연결 종료.")
                break

    except (socket.error, ConnectionRefusedError) as e:
        print(f"연결 오류: {e}. 서버에 재연결 시도 중...")
        time.sleep(5)  # 재시도 대기

    finally:
        client_socket.close()
        print("클라이언트 소켓 닫음.")