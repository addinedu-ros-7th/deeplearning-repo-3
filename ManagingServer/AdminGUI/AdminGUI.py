import cv2
import numpy as np
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# Socket setup
import socket

HOST = 'localhost'  # 서버의 호스트네임 또는 IP 주소
PORT = 12345        # 서버에서 사용하는 포트

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

def send_socket_message(message):
    try:
        client_socket.sendall(message.encode('utf-8'))
    except Exception as e:
        print(f"Error sending message '{message}': {e}")



# 목 중심 상대 좌표 변환 함수
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

# YOLOv8 Pose 모델 로드
pose_model = YOLO("yolov8m-pose.pt")

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

model = PoseClassifier(input_size=34, num_classes=2)
model.load_state_dict(torch.load("pose_classifier500.pth"))
model.eval()


# IoU 계산 함수 추가
def bbox_iou(boxA, boxB):
    # 두 박스의 교집합 영역 계산
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 각 박스의 영역 계산
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # IoU 계산
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# DeepSORT Tracker 초기화
tracker = DeepSort(max_age=5)

# 사람 상태를 추적하기 위한 딕셔너리
person_states = {}

# 웹캠 연결
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8로 키포인트 추출
    results = pose_model(frame)

    # Detection 목록 생성
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

    tracks = tracker.update_tracks(deep_sort_detections, frame=frame)

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
                    prediction = torch.argmax(model(input_tensor), dim=1).item()

                # 사람 상태 업데이트
                if track_id not in person_states:
                    person_states[track_id] = {
                        'detected': False,
                        #'in_region': False,
                        'posed': False
                    }
                    # "{personnumber} detected" 전송
                    send_socket_message(f"{track_id} detected")
                    person_states[track_id]['detected'] = True

                # 특정 지역에 대한 좌표 정의 (예: x1_region, y1_region, x2_region, y2_region)
                #x1_region, y1_region, x2_region, y2_region = 100, 100, 500, 500

                # 사람이 특정 지역에 있는지 확인
                #in_region = (x1 >= x1_region and y1 >= y1_region and x2 <= x2_region and y2 <= y2_region)

                # 지역 진입/이탈 상태 확인
                """
                if in_region and not person_states[track_id]['in_region']:
                    send_socket_message(f"{track_id} get in region")
                    person_states[track_id]['in_region'] = True
                elif not in_region and person_states[track_id]['in_region']:
                    send_socket_message(f"{track_id} get out region")
                    person_states[track_id]['in_region'] = False
                """
                
                # 타겟 포즈 여부 확인
                
                if prediction == 1 and not person_states[track_id]['posed']:
                    send_socket_message(f"{track_id} posed")
                    person_states[track_id]['posed'] = True
                elif prediction == 0 and person_states[track_id]['posed']:
                    person_states[track_id]['posed'] = False

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
    lost_track_ids = set(person_states.keys()) - current_track_ids
    for lost_track_id in lost_track_ids:
        send_socket_message(f"{lost_track_id} get out")
        del person_states[lost_track_id]

    # 화면 출력
    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
client_socket.close()
