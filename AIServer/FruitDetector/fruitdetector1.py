### File: cam1_server.py
import cv2
import time
import threading
from ultralytics import YOLO
from flask import Flask
from flask_socketio import SocketIO

# YOLO 모델 로드
model = YOLO("yolov8n.pt")

class SharedData:
    def __init__(self):
        self.lock = threading.Lock()
        self.detections_dict = {}

def yolo_detection_loop_cam1(shared_data, cap):
    """
    1번 카메라: 지속적으로 감지 결과를 갱신하고, 매 프레임마다 소켓으로 전송.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
        
        current_detections = {}
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            label = results[0].names[cls_id]
            current_detections[label] = current_detections.get(label, 0) + 1

        with shared_data.lock:
            shared_data.detections_dict = current_detections

        with shared_data.lock:
            data_to_send = shared_data.detections_dict.copy()
        socketio1.emit('yolo_data', data_to_send)

        display_frame = frame.copy()
        y_pos = 20
        for label, count in current_detections.items():
            cv2.putText(display_frame, f"{label}: {count}", (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            y_pos += 30
        cv2.imshow("Cam1 YOLO Detection Count", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Flask 앱 및 SocketIO 설정
app1 = Flask(__name__)
socketio1 = SocketIO(app1, cors_allowed_origins="*")

@app1.route('/')
def index1():
    return "Camera 1 YOLO detection server running on port 3000."

if __name__ == '__main__':
    shared_data_cam1 = SharedData()
    cap1 = cv2.VideoCapture(0)
    if not cap1.isOpened():
        print("Error: Camera 1 (index 0) could not be opened.")

    yolo_thread_cam1 = threading.Thread(target=yolo_detection_loop_cam1, args=(shared_data_cam1, cap1), daemon=True)
    yolo_thread_cam1.start()

    socketio1.run(app1, host='192.168.0.100', port=3000)