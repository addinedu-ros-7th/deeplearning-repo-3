import cv2
import time
import socket
import json
from threading import Thread, Lock
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("best.pt")

# 공유 데이터를 관리하는 클래스
class SharedData:
    def __init__(self):
        # 쓰레드 간 데이터 접근 동기화를 위한 Lock
        self.lock = Lock()
        # 감지된 데이터를 저장할 딕셔너리 {fruit_id: count}
        self.detections_dict = {}

# YOLO 감지 작업을 수행하는 쓰레드 클래스
class YOLOThread(Thread):
    def __init__(self, shared_data, cap):
        super().__init__(daemon=True)
        self.shared_data = shared_data  # 공유 데이터 객체
        self.cap = cap  # 카메라 객체
        self.running = True  # 쓰레드 실행 상태 플래그

    def run(self):
        while self.running:
            # 카메라로부터 프레임 읽기
            ret, frame = self.cap.read()
            if not ret:
                break

            # YOLO 모델을 사용하여 객체 감지 수행
            results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)

            # 감지 결과를 처리하여 딕셔너리에 저장
            current_detections = {}
            display_frame = frame.copy()
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())  # 클래스 ID
                label = results[0].names[cls_id]  # 클래스 이름
                fruit_id = self.parse_label(label)  # 클래스 이름을 fruit_id로 변환

                # 현재 감지된 항목 수 증가
                current_detections[fruit_id] = current_detections.get(fruit_id, 0) + 1

                # 바운딩 박스 정보
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                conf = box.conf[0].item()  # 신뢰도

                # 바운딩 박스와 라벨 그리기
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{label} ({conf:.2f})"
                cv2.putText(display_frame, label_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 공유 데이터를 업데이트 (Lock 사용)
            with self.shared_data.lock:
                self.shared_data.detections_dict = current_detections

            # 현재 감지 결과를 화면에 표시
            y_pos = 20
            for fruit_id, count in current_detections.items():
                label_text = f"Fruit ID {fruit_id}: {count}"
                cv2.putText(display_frame, label_text, (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30

            cv2.imshow("Cam2 YOLO Detection", display_frame)

            # 'q' 키를 눌러 쓰레드 중지
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()

        # 카메라 및 창 해제
        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        # 쓰레드 실행 상태 플래그 비활성화
        self.running = False

    def parse_label(self, label):
        # 클래스 이름을 fruit_id로 매핑
        # 요청한 형태에 맞추어 fruit_id를 단일 정수로 매핑합니다.
        label_mapping = {
            "apple_defective": 0, "apple_fair": 1,
            "mandarin_defective": 2, "mandarin_fair": 3,
            "peach_defective": 4, "peach_fair": 5,
            "pomegranate_defective": 6, "pomegranate_fair": 7
        }
        return label_mapping.get(label, -1)  # 매칭 안될 경우 -1

# 감지 데이터를 서버로 전송하는 쓰레드 클래스
class EmitThread(Thread):
    def __init__(self, shared_data, server_ip, server_port):
        super().__init__(daemon=True)
        self.shared_data = shared_data  # 공유 데이터 객체
        self.server_ip = server_ip  # 서버 IP 주소
        self.server_port = server_port  # 서버 포트 번호
        self.running = True  # 쓰레드 실행 상태 플래그
        self.connected = False  # 초기 연결 상태 플래그

    def connect_to_server(self):
        """서버에 연결을 시도하고 성공 여부를 반환"""
        while self.running and not self.connected:
            try:
                print("서버에 연결 시도 중...")
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.settimeout(5)  # 타임아웃 설정 (5초)
                self.client_socket.connect((self.server_ip, self.server_port))
                self.connected = True
                print(f"서버에 연결되었습니다. {self.server_ip}:{self.server_port}")
            except (socket.timeout, socket.error) as e:
                print(f"서버 연결 실패: {e}. 5초 후 재시도...")
                time.sleep(5)  # 5초 대기 후 재시도

    def run(self):
        # 서버 연결 시도
        self.connect_to_server()

        while self.running:
            # 연결이 끊어졌을 경우 재연결 시도
            if not self.connected:
                self.connect_to_server()

            # 연결된 상태에서 데이터 전송
            try:
                time.sleep(1)  # 10초 간격으로 데이터 전송

                with self.shared_data.lock:
                    # 공유 데이터를 JSON 형식으로 변환
                    data_to_send = {
                        "camera_id": "Fruit",
                        "data": [{"fruit_id": fruit_id, "stock": count}
                                 for fruit_id, count in self.shared_data.detections_dict.items()]
                    }
                self.client_socket.sendall(json.dumps(data_to_send).encode())
                print(f"데이터 전송: {data_to_send}")
            except (socket.error, BrokenPipeError) as e:
                print(f"데이터 전송 오류: {e}. 서버 연결 끊김.")
                self.connected = False  # 연결 상태 플래그를 False로 설정

    def stop(self):
        # 쓰레드 실행 상태 플래그 비활성화
        self.running = False

        if hasattr(self, "client_socket") and self.client_socket:
            self.client_socket.close()
            print("소켓이 닫혔습니다.")

if __name__ == '__main__':

    # 공유 데이터 객체 생성
    shared_data_cam2 = SharedData()

    # 카메라 초기화
    cap2 = cv2.VideoCapture(0)
    if not cap2.isOpened():
        print("Error: Camera 0 (index 0) could not be opened.")
        exit(1)

    # YOLO 감지 쓰레드 시작
    yolo_thread_cam2 = YOLOThread(shared_data_cam2, cap2)
    yolo_thread_cam2.start()

    # 서버 설정
    server_ip = '192.168.0.100'  # 서버 IP
    server_port = 5003           # 서버 포트

    # 데이터 전송 쓰레드 시작
    emit_thread_cam2 = EmitThread(shared_data_cam2, server_ip, server_port)
    emit_thread_cam2.start()

    try:
        # 메인 쓰레드 실행 유지
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # 프로그램 종료 시 쓰레드 중지
        print("프로그램 종료 중...")
    finally:
        yolo_thread_cam2.stop()
        emit_thread_cam2.stop()
        yolo_thread_cam2.join()
        emit_thread_cam2.join()
