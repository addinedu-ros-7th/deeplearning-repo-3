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
        # 감지된 데이터를 저장할 딕셔너리
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
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())  # 클래스 ID
                label = results[0].names[cls_id]  # 클래스 이름
                fruit_id, fair = self.parse_label(label)  # 클래스 이름을 fruit_id와 상태로 변환
                key = (fruit_id, fair)
                current_detections[key] = current_detections.get(key, 0) + 1

            # 공유 데이터를 업데이트 (Lock 사용)
            with self.shared_data.lock:
                self.shared_data.detections_dict = current_detections

            # 결과를 디스플레이에 표시
            display_frame = frame.copy()
            y_pos = 20
            for (fruit_id, fair), count in current_detections.items():
                fair_text = "fair" if fair else "defective"
                label_text = f"Fruit ID {fruit_id} ({fair_text}): {count}"
                cv2.putText(display_frame, label_text, (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                y_pos += 30
            cv2.imshow("Cam2 YOLO Detection Count", display_frame)

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
        # 클래스 이름을 fruit_id와 상태로 매핑
        label_mapping = {
            "apple_defective": (1, 0), "apple_fair": (1, 1),
            "peach_defective": (2, 0), "peach_fair": (2, 1),
            "pomegranate_defective": (3, 0), "pomegranate_fair": (3, 1),
            "mandarin_defective": (4, 0), "mandarin_fair": (4, 1)
        }
        return label_mapping.get(label, (0, 0))

# 감지 데이터를 서버로 전송하는 쓰레드 클래스
class EmitThread(Thread):
    def __init__(self, shared_data, server_ip, server_port):
        super().__init__(daemon=True)
        self.shared_data = shared_data  # 공유 데이터 객체
        self.server_ip = server_ip  # 서버 IP 주소
        self.server_port = server_port  # 서버 포트 번호
        self.running = True  # 쓰레드 실행 상태 플래그

    def run(self):
        while self.running:
            # 주기적으로 데이터 전송 (10초 간격)
            time.sleep(10)
            with self.shared_data.lock:
                # 공유 데이터를 JSON 형식으로 변환
                data_to_send = [
                    {"fruit_id": fruit_id, "fair": fair, "quantity": count} for (fruit_id, fair), count in self.shared_data.detections_dict.items()
                ]
            try:
                # 서버에 연결하여 데이터 전송
                print("시도")
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.connect((self.server_ip, self.server_port))
                    client_socket.sendall(json.dumps(data_to_send).encode())
                    print(f"데이터 전송: {data_to_send}")
            except (socket.error, ConnectionRefusedError) as e:
                print(f"소켓 오류 발생: {e}")

    def stop(self):
        # 쓰레드 실행 상태 플래그 비활성화
        self.running = False

if __name__ == '__main__':
    # 공유 데이터 객체 생성
    shared_data_cam2 = SharedData()

    # 카메라 초기화
    cap2 = cv2.VideoCapture(2)
    if not cap2.isOpened():
        print("Error: Camera 2 (index 2) could not be opened.")
        exit(1)

    # YOLO 감지 쓰레드 시작
    yolo_thread_cam2 = YOLOThread(shared_data_cam2, cap2)
    yolo_thread_cam2.start()

    # 서버 설정
    server_ip = '192.168.0.100'  # 서버 IP
    server_port = 5003       # 서버 포트

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
