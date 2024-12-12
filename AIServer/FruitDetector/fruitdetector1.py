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
        self.lock = Lock()
        self.detections_dict = {}

# YOLO 감지 작업을 수행하는 쓰레드 클래스
class YOLOThread(Thread):
    def __init__(self, shared_data, cap):
        super().__init__(daemon=True)
        self.shared_data = shared_data
        self.cap = cap
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)

            current_detections = {1: {}, 2: {}, 3: {}, 4: {}}  # 4개의 장바구니로 초기화

            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                label = results[0].names[cls_id]
                fruit_id, fair = self.parse_label(label)

                # 박스의 중심 좌표 계산
                x_center = (box.xywh[0][0].item() + box.xywh[0][2].item()) / 2

                # 위치를 기반으로 장바구니 할당
                cart_id = self.assign_cart(x_center)

                if cart_id:
                    key = (fruit_id, fair)
                    current_detections[cart_id][key] = current_detections[cart_id].get(key, 0) + 1

            with self.shared_data.lock:
                self.shared_data.detections_dict = current_detections

            display_frame = frame.copy()

            # 화면에 구역 나누기 시각화
            height, width, _ = display_frame.shape
            for i in range(1, 4):
                x = i * (width // 4)
                cv2.line(display_frame, (x, 0), (x, height), (255, 0, 0), 2)

            y_pos = 20
            for cart_id, fruits in current_detections.items():
                for (fruit_id, fair), count in fruits.items():
                    fair_text = "fair" if fair else "defective"
                    label_text = f"Cart {cart_id} - Fruit ID {fruit_id} ({fair_text}): {count}"
                    cv2.putText(display_frame, label_text, (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_pos += 30

            cv2.imshow("Cam2 YOLO Detection Count", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()

        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False

    def parse_label(self, label):
        label_mapping = {
            "apple_defective": (1, 0), "apple_fair": (1, 1),
            "peach_defective": (2, 0), "peach_fair": (2, 1),
            "pomegranate_defective": (3, 0), "pomegranate_fair": (3, 1),
            "mandarin_defective": (4, 0), "mandarin_fair": (4, 1)
        }
        return label_mapping.get(label, (0, 0))

    def assign_cart(self, x_center):
        print(x_center)
        if x_center < 160:
            return 1
        elif x_center < 220:
            return 2
        elif x_center < 300:
            return 3
        elif x_center < 360:
            return 4
        return None

# 감지 데이터를 서버로 전송하는 쓰레드 클래스
class EmitThread(Thread):
    def __init__(self, shared_data, server_ip, server_port):
        super().__init__(daemon=True)
        self.shared_data = shared_data
        self.server_ip = server_ip
        self.server_port = server_port
        self.running = True

    def run(self):
        while self.running:
            time.sleep(10)
            with self.shared_data.lock:
                data_to_send = {
                    "camera_id": "Cart",
                    "data": [
                        {
                            "cart_cam": cart_id,
                            "fruits": [
                                {fruit_id: count for (fruit_id, _), count in fruits.items()}
                            ]
                        }
                        for cart_id, fruits in self.shared_data.detections_dict.items()
                    ]
                }
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.connect((self.server_ip, self.server_port))
                    client_socket.sendall(json.dumps(data_to_send).encode())
                    print(f"데이터 전송: {data_to_send}")
            except (socket.error, ConnectionRefusedError) as e:
                print(f"소켓 오류 발생: {e}")

    def stop(self):
        self.running = False

if __name__ == '__main__':
    shared_data_cam2 = SharedData()

    cap2 = cv2.VideoCapture(2)
    if not cap2.isOpened():
        print("Error: Camera 2 (index 2) could not be opened.")
        exit(1)

    yolo_thread_cam2 = YOLOThread(shared_data_cam2, cap2)
    yolo_thread_cam2.start()

    server_ip = '192.168.0.100'
    server_port = 5002

    emit_thread_cam2 = EmitThread(shared_data_cam2, server_ip, server_port)
    emit_thread_cam2.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("프로그램 종료 중...")
    finally:
        yolo_thread_cam2.stop()
        emit_thread_cam2.stop()
        yolo_thread_cam2.join()
        emit_thread_cam2.join()
