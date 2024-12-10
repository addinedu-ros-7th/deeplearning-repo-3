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
    이 함수는 (17,3) 형태의 키포인트 배열(각 키포인트: [x, y, confidence])을 받아,
    목(neck)을 기준점으로 하는 상대 좌표계로 변환하고 어깨 폭을 스케일로 한 정규화를 수행합니다.
    여기서 keypoints_array[0]은 목, [5]는 왼어깨, [6]은 오른어깨 키포인트를 의미합니다.
    
    Steps:
    1. 목 위치를 origin으로 하여 전체 키포인트를 평행이동
    2. 왼어깨와 오른어깨 사이 거리(scale)를 구함
    3. 전체 키포인트를 scale로 나누어 정규화
    """
    neck = keypoints_array[0][:2]  # 목의 (x,y)
    if np.all(neck == 0):
        return None

    left_shoulder = keypoints_array[5][:2]
    right_shoulder = keypoints_array[6][:2]
    if np.all(left_shoulder == 0) or np.all(right_shoulder == 0):
        return None

    scale = np.linalg.norm(left_shoulder - right_shoulder)
    if scale == 0:
        return None

    relative_keypoints = keypoints_array[:, :2] - neck    # 목 기준으로 전체 키포인트 평행이동
    normalized_keypoints = relative_keypoints / scale      # 어깨 폭으로 스케일링
    return normalized_keypoints.flatten()                  # (17,2) -> (34,)로 flatten

def bbox_iou(boxA, boxB):
    """
    두 사각형(boxA, boxB)에 대한 IOU(Intersection Over Union) 값을 계산하는 함수.
    box는 [x1, y1, x2, y2] 형태로 주어지며,
    이는 이미지 상의 좌상단(x1,y1), 우하단(x2,y2) 좌표를 의미합니다.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # 두 박스의 교집합 영역 넓이
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 각 박스의 넓이
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # IOU 계산
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def convert_cv_qt(cv_img):
    """
    OpenCV의 BGR 이미지를 Qt에서 사용 가능한 QPixmap 형태로 변환하는 함수.
    """
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    p = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)

class PoseLSTMClassifier(nn.Module):
    """
    포즈(키포인트 시퀀스) 분류를 위한 LSTM 기반 모델 클래스.
    input_size: 한 프레임의 포즈 특징(키포인트) 차원 수
    hidden_size: LSTM 은닉 상태 크기
    num_layers: LSTM 레이어 수
    num_classes: 최종 분류 클래스 수
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(PoseLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM 레이어
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 드롭아웃
        self.dropout = nn.Dropout(p=0.5)
        # 완전연결 계층
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # 초기 은닉 상태와 셀 상태 정의
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 통과
        out, _ = self.lstm(x, (h_0, c_0))
        
        # 시퀀스 마지막 타임스텝의 출력을 사용
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class VideoThread(QThread):
    """
    QThread를 상속한 비디오 처리 스레드.
    이 스레드는 웹캠에서 프레임을 읽어, YOLOv8 pose 모델로 사람을 검출하고,
    DeepSORT로 추적하며, LSTM 모델을 통해 포즈를 분류한 후,
    비디오에 해당 정보(포즈, ID)를 표시합니다.
    또한 일정 시간(1분)마다 비디오 파일을 롤오버(새 파일로 교체) 저장합니다.
    """
    # PyQt 시그널: 메인 스레드와 통신
    change_pixmap_signal = pyqtSignal(np.ndarray)  # 이미지 업데이트 시그널
    person_entered = pyqtSignal(int)               # 새로운 사람 진입 시그널
    person_posed = pyqtSignal(int)                 # 사람이 특정 포즈를 취하기 시작했을 때 시그널
    person_stopped_posing = pyqtSignal(int)        # 사람이 특정 포즈를 멈췄을 때 시그널
    person_exited = pyqtSignal(int)                # 사람이 프레임에서 사라질 때 시그널
    file_path = pyqtSignal(str)                    # 저장 파일 경로 변경 시 시그널

    def __init__(self):
        super().__init__()
        self._run_flag = True

        # YOLOv8 Pose 모델 로드 (yolov8m-pose 모델)
        self.pose_model = YOLO("yolov8m-pose.pt", verbose=False)

        # LSTM 모델 로드
        input_size = 34    # 17개 키포인트 * 2차원 = 34
        hidden_size = 128
        num_layers = 2
        num_classes = 2    # 예: Normal Pose, Target Pose 2가지 분류
        self.model = PoseLSTMClassifier(input_size, hidden_size, num_layers, num_classes)
        self.model.load_state_dict(torch.load("pose_lstm_classifier.pth"))
        self.model.eval()

        # DeepSORT 추적기 초기화
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

        # 포즈 시퀀스 버퍼 (LSTM 입력용)
        self.sequence_length = 10
        self.sequence_buffers = {}  # track_id 별로 최근 포즈 시퀀스를 저장
        self.prev_poses = {}        # track_id 별 이전 포즈 상태(0: Normal, 1: Target)
        self.prev_track_ids = set() # 이전 프레임에 존재하던 track_id 집합

    def run(self):
        # 웹캠(카메라) 열기
        cap = cv2.VideoCapture(0)

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break

            # 비디오 저장용 VideoWriter 초기화 (첫 프레임에서)
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
                # 결과를 시각화한 프레임
                annotated_frame = results[0].plot()

            # 첫번째 결과(가장 가능성 높은 frame 결과) 추출
            result = results[0] if len(results) > 0 else None

            detections = []
            # 키포인트 정보 추출 (각 사람 별 박스, 신뢰도, 키포인트)
            if result is not None and result.keypoints is not None and result.keypoints.xy is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                kpts = result.keypoints.xy.cpu().numpy()  # (N,17,3)
                for i, (box, confidence) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = box
                    detection_keypoints = kpts[i]  # i번째 인물의 17개 키포인트
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'keypoints': detection_keypoints
                    })

            # DeepSORT를 이용한 추적 업데이트
            deep_sort_detections = [(d['bbox'], d['confidence']) for d in detections]
            tracks = self.tracker.update_tracks(deep_sort_detections, frame=frame)

            current_track_ids = set()

            for track in tracks:
                # 확정된 트랙만 처리
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                track_bbox = track.to_tlbr()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = track_bbox
                current_track_ids.add(track_id)

                # 새롭게 등장한 track_id는 이벤트 발신 (person_entered)
                if track_id not in self.prev_track_ids:
                    self.person_entered.emit(track_id)
                    self.sequence_buffers[track_id] = collections.deque(maxlen=self.sequence_length)

                # 추적된 박스와 detection 중 가장 높은 IOU를 가지는 detection 찾기
                best_iou = 0
                best_detection = None
                for d in detections:
                    iou = bbox_iou(track_bbox, d['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_detection = d

                # 해당 track_id와 매칭되는 best_detection이 있으면 포즈 분석
                if best_detection is not None:
                    kp_array = best_detection['keypoints']  # (17,3)
                    relative_keypoints = convert_to_neck_relative_coordinates(kp_array)

                    if relative_keypoints is not None: # 키포인트가 유효한 값이 맞다면
                        # 시퀀스 버퍼에 현재 프레임의 정규화 키포인트 추가
                        self.sequence_buffers[track_id].append(relative_keypoints)

                        # 시퀀스 길이 충족 시 LSTM에 넣어 분류
                        if len(self.sequence_buffers[track_id]) == self.sequence_length:
                            input_sequence = np.array(self.sequence_buffers[track_id])  
                            # (sequence_length, 34) 형태
                            input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 34)
                            
                            with torch.no_grad():
                                outputs = self.model(input_tensor)
                                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                                predicted = np.argmax(probabilities)
                                prediction = predicted.item()

                            # 이전 포즈 상태와 비교해 이벤트 발생
                            prev_pose = self.prev_poses.get(track_id, 0)
                            # 예: 0->1로 변경 시 person_posed 이벤트
                            if prediction == 1 and prev_pose == 0:
                                self.person_posed.emit(track_id)
                            # 1->0으로 변경 시 person_stopped_posing 이벤트
                            elif prediction == 0 and prev_pose == 1:
                                self.person_stopped_posing.emit(track_id)

                            self.prev_poses[track_id] = prediction

                            # 프레임에 포즈 상태 표시
                            pose_texts = ["Normal Pose", "Target Pose"]
                            colors = [(0, 0, 255), (0, 255, 0)]
                            cv2.putText(annotated_frame, pose_texts[prediction], (int(x1), int(y2)+20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[prediction], 2)
                        else:
                            # 아직 LSTM용 시퀀스가 충분하지 않다면 기본 Normal Pose로 가정
                            self.prev_poses[track_id] = 0
                            cv2.putText(annotated_frame, "Normal Pose", (int(x1), int(y2)+20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    else:
                        # 키포인트가 유효한 값이 아니라면 Normal Pose
                        self.prev_poses[track_id] = 0
                        cv2.putText(annotated_frame, "Normal Pose", (int(x1), int(y2)+20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                else:
                    # 해당 track_id에 매칭되는 detection이 없으면 Normal Pose
                    self.prev_poses[track_id] = 0
                    cv2.putText(annotated_frame, "Normal Pose", (int(x1), int(y2)+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                # 영상에 track_id 표시
                cv2.putText(annotated_frame, f"ID: {track_id}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # 이전에 있던 트랙 중 현재 프레임에 없는 트랙은 퇴장 처리
            lost_track_ids = self.prev_track_ids - current_track_ids
            for lost_track_id in lost_track_ids:
                self.person_exited.emit(lost_track_id)
                if lost_track_id in self.prev_poses:
                    del self.prev_poses[lost_track_id]
                if lost_track_id in self.sequence_buffers:
                    del self.sequence_buffers[lost_track_id]

            self.prev_track_ids = current_track_ids.copy()

            # 비디오 프레임 저장
            if annotated_frame is not None and self.video_writer is not None:
                self.video_writer.write(annotated_frame)

            # 1분 간격으로 비디오 파일 롤오버(새로 저장 시작)
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

            # UI 업데이트를 위해 시그널 발행
            if annotated_frame is not None:
                self.change_pixmap_signal.emit(annotated_frame)

            if not self._run_flag:
                break

        cap.release()
        # 스레드 종료 시 비디오 파일 정리
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            import os
            os.rename(self.current_video_save_path, self.future_video_save_path)

    def stop(self):
        # 스레드 정지 요청
        self._run_flag = False
        self.wait()
