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

# 유틸리티 함수 정의
def convert_to_neck_relative_coordinates(keypoints):
    """
    키포인트 데이터를 목(neck) 중심 좌표로 변환하고 스케일 정규화하는 함수.
    keypoints: Keypoints 객체.
    """
    keypoints_array = keypoints.xy.cpu().numpy()  # GPU -> CPU 변환 및 numpy로 변환

    # 최소한 필요한 키포인트 수 확인 (17개)
    if keypoints_array.shape[0] < 7:  # 목(0), 어깨(5, 6) 포함 최소 7개가 필요
        return None  # 유효하지 않은 데이터 처리

    neck = keypoints_array[0][:2]  # 목 좌표 가져오기

    if np.all(neck == 0):
        return None  # 목 좌표가 없는 경우 무효 처리

    # 어깨 좌표 가져오기
    left_shoulder = keypoints_array[5][:2]
    right_shoulder = keypoints_array[6][:2]

    # 어깨 좌표 유효성 검사
    if np.all(left_shoulder == 0) or np.all(right_shoulder == 0):
        return None  # 어깨 좌표가 없는 경우 무효 처리

    # 스케일 계산 (어깨 간 거리)
    scale = np.linalg.norm(left_shoulder - right_shoulder)
    if scale == 0:
        return None  # 스케일이 0인 경우 무효 처리

    # 목 기준 상대 좌표 변환 (평행 이동)
    relative_keypoints = keypoints_array[:, :2] - neck

    # 스케일 정규화
    normalized_keypoints = relative_keypoints / scale

    return normalized_keypoints.flatten()


def bbox_iou(boxA, boxB):
    """
    두 바운딩 박스의 IoU (Intersection over Union) 계산.
    boxA, boxB: [x1, y1, x2, y2] 형태의 리스트.
    """
    # 교차 영역 계산
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 각 박스의 면적 계산
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # IoU 계산
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)  # 0으로 나누는 것을 방지
    return iou

def convert_cv_qt(cv_img):
    """
    OpenCV 이미지를 QPixmap으로 변환.
    """
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    p = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)

# 비디오 처리 쓰레드 클래스 정의
class VideoThread(QThread):
    # PyQt 시그널 정의
    change_pixmap_signal = pyqtSignal(np.ndarray)  # 프레임 전송 신호
    person_entered = pyqtSignal(int)  # 새로운 사람이 화면에 등장했을 때
    person_posed = pyqtSignal(int)  # 사람이 특정 포즈를 취했을 때
    person_stopped_posing = pyqtSignal(int)  # 사람이 포즈를 멈췄을 때
    person_exited = pyqtSignal(int)  # 사람이 화면에서 사라졌을 때
    file_path = pyqtSignal(str) #비디오가 저장되는 경로

    def __init__(self):
        super().__init__()
        self._run_flag = True  # 쓰레드 실행 상태 플래그

        # YOLOv8 Pose 모델 로드
        self.pose_model = YOLO("yolov8m-pose.pt", verbose=False)

        # 포즈 분류 모델 정의 및 초기화
        class PoseLSTMClassifier(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, num_classes):
                super(PoseLSTMClassifier, self).__init__()
                self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.dropout = torch.nn.Dropout(p=0.5)
                self.fc1 = torch.nn.Linear(hidden_size, 64)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(64, num_classes)

            def forward(self, x):
                h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
                c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
                out, _ = self.lstm(x, (h_0, c_0))
                out = out[:, -1, :]
                out = self.dropout(out)
                out = self.fc1(out)
                out = self.relu(out)
                out = self.fc2(out)
                return out  # 로짓(logits) 반환

        input_size = 34  # 17 키포인트 x 2 좌표
        hidden_size = 128
        num_layers = 2
        num_classes = 2  # 클래스 수

        self.model = PoseLSTMClassifier(input_size, hidden_size, num_layers, num_classes)
        self.model.load_state_dict(torch.load("pose_lstm_classifier.pth"))
        self.model.eval()  # 평가 모드로 설정

        # DeepSORT Tracker 초기화
        self.tracker = DeepSort(max_age=5)  # 추적 상태가 유지될 최대 프레임 수 설정

        # 영상 저장 관련 초기화
        self.video_writer = None  # 초기화
        self.current_video_save_path = "./video_output/output.avi"  # 임시 파일 경로
        timestamp = (datetime.now() + timedelta(minutes=1)).strftime("%Y%m%d_%H%M%S")
        self.future_video_save_path = f"./video_output/output_{timestamp}.avi"  # 앞으로 저장될 파일 경로
        self.fps = 30
        self.frame_size = None
        self.last_save_time = datetime.now()

        # 저장될 비디오의 경로를 시그널로 전송
        self.file_path.emit(self.future_video_save_path)

        # 시퀀스 버퍼 설정
        self.sequence_length = 10  # 시퀀스 길이
        self.sequence_buffers = {}  # 각 트랙 ID에 대한 시퀀스 버퍼 저장

    def run(self):
        # 웹캠 비디오 캡처
        cap = cv2.VideoCapture(0)
        prev_track_ids = set()  # 이전 프레임에서 추적된 ID
        prev_poses = {}  # 이전 프레임에서 각 ID의 포즈 상태

        while self._run_flag:  # 쓰레드가 실행 중일 동안 반복
            ret, frame = cap.read()
            if not ret:  # 프레임을 읽을 수 없으면 종료
                break

            # 비디오 저장 초기화
            if self.video_writer is None:
                self.frame_size = (frame.shape[1], frame.shape[0])
                self.video_writer = cv2.VideoWriter(
                    self.current_video_save_path,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    self.fps,
                    self.frame_size
                )

            # YOLOv8로 키포인트 추출
            results = self.pose_model(frame, verbose=False)

            # Detection 목록 생성
            detections = []
            for result in results:
                if result.keypoints is None:
                    print("키포인트가 감지되지 않음")
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표
                confidences = result.boxes.conf.cpu().numpy()  # 각 바운딩 박스의 정확도
                keypoints = result.keypoints  # 키포인트 데이터

                for box, confidence, kp in zip(boxes, confidences, keypoints):
                    x1, y1, x2, y2 = box
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'keypoints': kp  # Keypoints 객체
                    })

            # DeepSORT에 Detection 전달
            deep_sort_detections = []
            for detection in detections:
                bbox = detection['bbox']
                confidence = detection['confidence']
                deep_sort_detections.append((bbox, confidence))

            # 객체 추적 정보 업데이트
            tracks = self.tracker.update_tracks(deep_sort_detections, frame=frame)

            # 현재 프레임의 추적된 객체 ID 목록
            current_track_ids = set()

            for track in tracks:
                if not track.is_confirmed():  # 추적이 확인되지 않은 경우 무시
                    continue
                track_id = track.track_id  # 추적 ID
                track_bbox = track.to_tlbr()  # 바운딩 박스 좌표 [x1, y1, x2, y2]
                x1, y1, x2, y2 = track_bbox
                current_track_ids.add(track_id)

                # 새로운 트랙 ID 감지
                if track_id not in prev_track_ids:
                    self.person_entered.emit(track_id)  # 새로운 사람이 등장했다고 신호 전송
                    # 시퀀스 버퍼 초기화
                    self.sequence_buffers[track_id] = collections.deque(maxlen=self.sequence_length)

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
                    keypoints = best_detection['keypoints']  # 키포인트 데이터
                    try:
                        # 목 중심 상대 좌표 변환
                        relative_keypoints = convert_to_neck_relative_coordinates(keypoints)

                        if relative_keypoints is not None:
                            # 시퀀스 버퍼에 추가
                            self.sequence_buffers[track_id].append(relative_keypoints)

                            # 충분한 프레임이 모였는지 확인
                            if len(self.sequence_buffers[track_id]) == self.sequence_length:
                                input_sequence = np.array(self.sequence_buffers[track_id])  # 형상: (sequence_length, input_size)
                                input_tensor = torch.tensor([input_sequence], dtype=torch.float32)  # 배치 차원 추가

                                # 예측
                                with torch.no_grad():
                                    outputs = self.model(input_tensor)
                                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                                    predicted = np.argmax(probabilities)
                                    prediction = predicted.item()

                                # 이전 포즈와 비교
                                prev_pose = prev_poses.get(track_id, 0)
                                if prediction == 1 and prev_pose == 0:
                                    self.person_posed.emit(track_id)  # 포즈 시작 신호 전송
                                elif prediction == 0 and prev_pose == 1:
                                    self.person_stopped_posing.emit(track_id)  # 포즈 중지 신호 전송

                                # 이전 포즈 상태 업데이트
                                prev_poses[track_id] = prediction

                                # 결과 표시
                                pose_texts = ["Normal Pose", "Target Pose"]
                                colors = [(0, 0, 255), (0, 255, 0)]  # Red for normal, green for target

                                cv2.putText(frame, pose_texts[prediction], (int(x1), int(y2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[prediction], 2)
                        else:
                            # 유효하지 않은 키포인트는 일반 포즈로 처리
                            prediction = 0  # Normal Pose
                            prev_poses[track_id] = prediction
                            cv2.putText(frame, "Normal Pose", (int(x1), int(y2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        # 바운딩 박스와 ID 표시
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    except Exception as e:
                        print(f"트랙 {track_id}의 키포인트 처리 중 오류: {e}")
                else:
                    print(f"트랙 {track_id}에 매칭되는 Detection이 없음")

            # 사라진 트랙 감지
            lost_track_ids = prev_track_ids - current_track_ids
            for lost_track_id in lost_track_ids:
                self.person_exited.emit(lost_track_id)  # 사람이 사라졌다는 신호 전송
                if lost_track_id in prev_poses:
                    del prev_poses[lost_track_id]  # 이전 포즈 상태에서 삭제
                if lost_track_id in self.sequence_buffers:
                    del self.sequence_buffers[lost_track_id]  # 시퀀스 버퍼 삭제

            # 이전 프레임 ID 목록 업데이트
            prev_track_ids = current_track_ids.copy()
            
            # 비디오 저장
            if frame is not None and self.video_writer is not None:
                self.video_writer.write(frame)

            # 1분 경과 확인 후 저장 완료
            if datetime.now() - self.last_save_time >= timedelta(minutes=1):
                # 비디오 저장 종료
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None

                # 임시 파일을 미래 경로로 이름 변경
                import os
                os.rename(self.current_video_save_path, self.future_video_save_path)

                # 다음 비디오 저장 경로 계산
                self.last_save_time = datetime.now()
                timestamp = (self.last_save_time + timedelta(minutes=1)).strftime("%Y%m%d_%H%M%S")
                self.future_video_save_path = f"./video_output/output_{timestamp}.avi"

                # 다음에 저장될 비디오 경로를 시그널로 전송
                self.file_path.emit(self.future_video_save_path)

                # 비디오 저장 초기화 (다시 임시 파일로 시작)
                self.video_writer = cv2.VideoWriter(
                    self.current_video_save_path,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    self.fps,
                    self.frame_size
                )

            # cv2.imshow 대신 프레임을 시그널로 전송
            self.change_pixmap_signal.emit(frame)

            # 종료 플래그 확인
            if not self._run_flag:
                break

        cap.release()  # 비디오 캡처 해제
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

            # 마지막으로 저장된 비디오를 미래 경로로 이름 변경
            import os
            os.rename(self.current_video_save_path, self.future_video_save_path)

    def stop(self):
        """
        쓰레드 종료 함수.
        """
        self._run_flag = False
        self.wait()
