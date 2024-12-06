import sys
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import collections
from deep_sort_realtime.deepsort_tracker import DeepSort
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from datetime import datetime, timedelta
import torch.nn as nn

def convert_to_neck_relative_coordinates(keypoints_array):
    """
    keypoints_array: (17,3) 형태의 numpy 배열 [x, y, confidence]
    목(0), 왼어깨(5), 오른어깨(6)를 이용해 neck 기준 상대좌표로 변환하고 스케일 정규화.
    """
    neck = keypoints_array[0][:2]

    if np.all(neck == 0):
        return None

    left_shoulder = keypoints_array[5][:2]
    right_shoulder = keypoints_array[6][:2]

    if np.all(left_shoulder == 0) or np.all(right_shoulder == 0):
        return None

    scale = np.linalg.norm(left_shoulder - right_shoulder)
    if scale == 0:
        return None

    relative_keypoints = keypoints_array[:, :2] - neck
    normalized_keypoints = relative_keypoints / scale

    return normalized_keypoints.flatten()

def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def convert_cv_qt(cv_img):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    p = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)

class PoseLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(PoseLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    person_entered = pyqtSignal(int)
    person_posed = pyqtSignal(int)
    person_stopped_posing = pyqtSignal(int)
    person_exited = pyqtSignal(int)
    file_path = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True

        # YOLOv8 Pose 모델 로드
        self.pose_model = YOLO("yolov8m-pose.pt", verbose=False)

        # LSTM 모델 로드
        input_size = 34
        hidden_size = 128
        num_layers = 2
        num_classes = 2
        self.model = PoseLSTMClassifier(input_size, hidden_size, num_layers, num_classes)
        self.model.load_state_dict(torch.load("pose_lstm_classifier.pth"))
        self.model.eval()

        self.tracker = DeepSort(max_age=5)

        # 비디오 저장 초기화
        self.video_writer = None
        self.current_video_save_path = "./video_output/output.avi"
        timestamp = (datetime.now() + timedelta(minutes=1)).strftime("%Y%m%d_%H%M%S")
        self.future_video_save_path = f"./video_output/output_{timestamp}.avi"
        self.fps = 30
        self.frame_size = None
        self.last_save_time = datetime.now()
        self.file_path.emit(self.future_video_save_path)

        # 시퀀스 버퍼
        self.sequence_length = 10
        self.sequence_buffers = {}
        self.prev_poses = {}
        self.prev_track_ids = set()

    def run(self):
        cap = cv2.VideoCapture(0)

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break

            if self.video_writer is None:
                self.frame_size = (frame.shape[1], frame.shape[0])
                self.video_writer = cv2.VideoWriter(
                    self.current_video_save_path,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    self.fps,
                    self.frame_size
                )

            # YOLOv8 Pose 추론
            results = self.pose_model(frame, verbose=False)
            if len(results) == 0:
                annotated_frame = frame.copy()
            else:
                annotated_frame = results[0].plot()

            result = results[0] if len(results) > 0 else None
            detections = []

            if result is not None and result.keypoints is not None and result.keypoints.xy is not None:
                # result.keypoints.xy: shape (N, 17, 3)
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                kpts = result.keypoints.xy.cpu().numpy()  # (N,17,3) 형태, 모든 디텍션 키포인트

                for i, (box, confidence) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = box
                    detection_keypoints = kpts[i]  # i번째 디텍션의 키포인트 (17,3)
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'keypoints': detection_keypoints
                    })

            # DeepSORT 추적
            deep_sort_detections = [(d['bbox'], d['confidence']) for d in detections]
            tracks = self.tracker.update_tracks(deep_sort_detections, frame=frame)

            current_track_ids = set()

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                track_bbox = track.to_tlbr()
                x1, y1, x2, y2 = track_bbox
                current_track_ids.add(track_id)

                if track_id not in self.prev_track_ids:
                    self.person_entered.emit(track_id)
                    self.sequence_buffers[track_id] = collections.deque(maxlen=self.sequence_length)

                # best_iou 매칭
                best_iou = 0
                best_detection = None
                for d in detections:
                    iou = bbox_iou(track_bbox, d['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_detection = d

                if best_detection is not None:
                    kp_array = best_detection['keypoints']  # (17,3)
                    relative_keypoints = convert_to_neck_relative_coordinates(kp_array)

                    if relative_keypoints is not None:
                        self.sequence_buffers[track_id].append(relative_keypoints)

                        if len(self.sequence_buffers[track_id]) == self.sequence_length:
                            input_sequence = np.array(self.sequence_buffers[track_id])  # (sequence_length, feature_dim) 형태의 numpy 배열
                            input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)

                            with torch.no_grad():
                                outputs = self.model(input_tensor)
                                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                                predicted = np.argmax(probabilities)
                                prediction = predicted.item()

                            prev_pose = self.prev_poses.get(track_id, 0)
                            if prediction == 1 and prev_pose == 0:
                                self.person_posed.emit(track_id)
                            elif prediction == 0 and prev_pose == 1:
                                self.person_stopped_posing.emit(track_id)

                            self.prev_poses[track_id] = prediction

                            pose_texts = ["Normal Pose", "Target Pose"]
                            colors = [(0, 0, 255), (0, 255, 0)]
                            cv2.putText(annotated_frame, pose_texts[prediction], (int(x1), int(y2)+20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[prediction], 2)
                        else:
                            # 아직 시퀀스가 다 안 모임 -> Normal Pose 가정
                            self.prev_poses[track_id] = 0
                            cv2.putText(annotated_frame, "Normal Pose", (int(x1), int(y2)+20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    else:
                        # 유효하지 않은 키포인트 -> Normal Pose
                        self.prev_poses[track_id] = 0
                        cv2.putText(annotated_frame, "Normal Pose", (int(x1), int(y2)+20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                else:
                    # Detection이 없으므로 Normal Pose
                    self.prev_poses[track_id] = 0
                    cv2.putText(annotated_frame, "Normal Pose", (int(x1), int(y2)+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                # ID 표시
                cv2.putText(annotated_frame, f"ID: {track_id}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # 사라진 트랙 처리
            lost_track_ids = self.prev_track_ids - current_track_ids
            for lost_track_id in lost_track_ids:
                self.person_exited.emit(lost_track_id)
                if lost_track_id in self.prev_poses:
                    del self.prev_poses[lost_track_id]
                if lost_track_id in self.sequence_buffers:
                    del self.sequence_buffers[lost_track_id]

            self.prev_track_ids = current_track_ids.copy()

            # 비디오 저장
            if annotated_frame is not None and self.video_writer is not None:
                self.video_writer.write(annotated_frame)

            # 1분 경과 후 비디오 롤오버
            if datetime.now() - self.last_save_time >= timedelta(minutes=1):
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None

                import os
                os.rename(self.current_video_save_path, self.future_video_save_path)

                self.last_save_time = datetime.now()
                timestamp = (self.last_save_time + timedelta(minutes=1)).strftime("%Y%m%d_%H%M%S")
                self.future_video_save_path = f"./video_output/output_{timestamp}.avi"
                self.file_path.emit(self.future_video_save_path)

                self.video_writer = cv2.VideoWriter(
                    self.current_video_save_path,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    self.fps,
                    self.frame_size
                )

            if annotated_frame is not None:
                self.change_pixmap_signal.emit(annotated_frame)

            if not self._run_flag:
                break

        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            import os
            os.rename(self.current_video_save_path, self.future_video_save_path)

    def stop(self):
        self._run_flag = False
        self.wait()
