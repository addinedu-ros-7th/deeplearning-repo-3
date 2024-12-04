import sys
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import DBConnector

from_class = uic.loadUiType("AdminGUI.ui")[0]

# Define the Person class
class Person:
    def __init__(self, track_id):
        self.track_id = track_id
        self.detected = False
        self.posed = False

# Define utility functions
def convert_to_neck_relative_coordinates(keypoints):
    """
    Converts keypoints to neck-relative coordinates.
    keypoints: numpy array of shape (num_keypoints, 3)
    """
    # Extract x, y coordinates
    keypoints_xy = keypoints[:, :2]  # Shape: (num_keypoints, 2)
    
    # Get neck coordinates (keypoint index 0)
    neck_x, neck_y = keypoints_xy[0]

    # Check for missing neck keypoint
    if neck_x == 0 and neck_y == 0:
        return None  # Invalid data

    # Get wrist coordinates (left wrist index 9, right wrist index 10)
    left_wrist_x, left_wrist_y = keypoints_xy[9]
    right_wrist_x, right_wrist_y = keypoints_xy[10]

    # Check for missing wrist keypoints
    if (left_wrist_x == 0 and left_wrist_y == 0) or (right_wrist_x == 0 and right_wrist_y == 0):
        return None  # Invalid data

    # Compute relative keypoints
    relative_keypoints = keypoints_xy - np.array([neck_x, neck_y])

    # Flatten to a 1D array
    return relative_keypoints.flatten()

def bbox_iou(boxA, boxB):
    # Calculate the intersection area
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calculate the area of both boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Calculate IoU
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def convert_cv_qt(cv_img):
    """Convert from an OpenCV image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    p = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)

# Video processing thread
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

        # YOLOv8 Pose 모델 로드
        self.pose_model = YOLO("yolov8m-pose.pt")

        # 딥러닝 모델 로드
        class PoseClassifier(torch.nn.Module):
            def __init__(self, input_size, num_classes):
                super(PoseClassifier, self).__init__()
                self.fc = torch.nn.Sequential(
                    torch.nn.Linear(input_size, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, num_classes)
                )
            
            def forward(self, x):
                return self.fc(x)

        self.model = PoseClassifier(input_size=34, num_classes=2)
        self.model.load_state_dict(torch.load("pose_classifier500.pth"))
        self.model.eval()

        # DeepSORT Tracker 초기화
        self.tracker = DeepSort(max_age=5)

        # 사람 상태를 추적하기 위한 딕셔너리
        self.person_states = {}

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLOv8로 키포인트 추출
            results = self.pose_model(frame)

            # Detection 목록 생성 (모든 디텍팅된 개체들 detection에 담음)
            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                keypoints_tensor = result.keypoints.data.cpu()  # Tensor of shape (num_persons, num_keypoints, 3)
                keypoints_list = keypoints_tensor.numpy()

                for box, confidence, keypoints in zip(boxes, confidences, keypoints_list):
                    x1, y1, x2, y2 = box
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'keypoints': keypoints  # NumPy array of shape (num_keypoints, 3)
                    })

            # DeepSORT에 Detection 전달
            deep_sort_detections = []
            for detection in detections:
                bbox = detection['bbox']
                confidence = detection['confidence']
                deep_sort_detections.append((bbox, confidence))

            tracks = self.tracker.update_tracks(deep_sort_detections, frame=frame)

            # 현재 프레임에서 추적된 트랙의 ID 목록
            current_track_ids = set()

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id  # personnumber
                track_bbox = track.to_tlbr()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = track_bbox
                current_track_ids.add(track_id)

                # 가장 매칭되는 Detection 찾기
                best_iou = 0
                best_detection = None
                for detection in detections:
                    detection_bbox = detection['bbox']
                    iou = bbox_iou(track_bbox, detection_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_detection = detection

                if best_detection is not None:
                    keypoints = best_detection['keypoints']  # NumPy array of shape (num_keypoints, 3)
                    try:
                        # 목 중심 상대 좌표 변환
                        relative_keypoints = convert_to_neck_relative_coordinates(keypoints)

                        # 유효하지 않은 키포인트는 일반 포즈로 취급
                        if relative_keypoints is None:
                            prediction = 0  # 일반 포즈
                        else:
                            # 딥러닝 모델로 예측
                            input_tensor = torch.tensor([relative_keypoints], dtype=torch.float32)
                            prediction = torch.argmax(self.model(input_tensor), dim=1).item()

                        # 사람 상태 업데이트
                        if track_id not in self.person_states:
                            self.person_states[track_id] = Person(track_id)
                            self.person_states[track_id].detected = True

                        # 타겟 포즈 여부 확인
                        if prediction == 1 and not self.person_states[track_id].posed:
                            # 사람이 포즈를 취했을 때 수행할 동작 
                            # send_socket_message(f"{track_id} posed")
                            self.person_states[track_id].posed = True
                        elif prediction == 0 and self.person_states[track_id].posed:
                            self.person_states[track_id].posed = False

                        # 결과 표시
                        if prediction == 1:
                            cv2.putText(frame, "Target Pose", (int(x1), int(y2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "Normal Pose", (int(x1), int(y2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        # 바운딩 박스 및 ID 표시
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    except Exception as e:
                        print(f"Error processing keypoints for track {track_id}: {e}")
                else:
                    print(f"No matching detection for track {track_id}")

            # 사라진 트랙 감지 및 처리
            lost_track_ids = set(self.person_states.keys()) - current_track_ids
            for lost_track_id in lost_track_ids:
                # 사람이 나갔을 때 수행할 동작
                # send_socket_message(f"{lost_track_id} get out")
                del self.person_states[lost_track_id]

            # Instead of cv2.imshow, emit the frame
            self.change_pixmap_signal.emit(frame)

            # Optional: Add a small sleep to prevent overloading the CPU
            # time.sleep(0.01)

            # Check for stop flag
            if not self._run_flag:
                break

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# Main window class
class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Detection")
        self.setupUi(self)

        # Start the video thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def convert_cv_qt(self, cv_img, label_width, label_height):
        """Convert from an OpenCV image to QPixmap and scale it to the label size"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # 
        qt_image = qt_image.scaled(label_width, label_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(qt_image)


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Update the image_label with a new OpenCV image"""
        label1_width = self.CCTV_1.width()
        label1_height = self.CCTV_1.height()
        label2_width = self.CCTV_2.width()
        label2_height = self.CCTV_2.height()

        qt_img1 = self.convert_cv_qt(cv_img, label1_width, label1_height)
        qt_img2 = self.convert_cv_qt(cv_img, label2_width, label2_height)

        self.CCTV_1.setPixmap(qt_img1)
        self.CCTV_2.setPixmap(qt_img2)


    def closeEvent(self, event):
        """Stop the video thread when the window is closed"""
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())
