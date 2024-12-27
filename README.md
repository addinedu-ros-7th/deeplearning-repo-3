<h1>:apple: Fruit-Flow-Market :apple:</h1>
<br>
<h2>Project Goals</h2>
<b>무인 스마트 매장 시스템</b>을 구현합니다.<br>

1. 얼굴 인식 자동 결제<br>
얼굴 인식을 통해 자동으로 결제하여 고객의 상품 구매 경험을 최적화합니다.

2. 도난 행위 감지<br>
딥러닝 기술 기반 매장 내 도난 행위를 감지하여 안전한 쇼핑 환경을 보장합니다.

3. 매장 및 과일 상태 모니터링<br>
매장 환경 및 과일 신선도 모니터링을 통해 효율적인 재고 관리 및 운영을 지원합니다.

<img src="https://github.com/user-attachments/assets/8a4c2b58-3dfa-403b-a968-953defcad32c" width="420">

<br>
<br>
<br>
<h2>Tech Stack</h2>
<br>
<div align="center">
  <img src="https://img.shields.io/badge/Python 3.10-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/CUDA 12.4-76B900?style=for-the-badge&logo=nvidia&logoColor=white">
  <img src="https://img.shields.io/badge/Ubuntu 22.04-E95420?style=for-the-badge&logo=ubuntu&logoColor=white">
  <br>
  <img src="https://img.shields.io/badge/PyTorch 2.5.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white">
  <img src="https://img.shields.io/badge/PyQt-41CD52?style=for-the-badge&logo=qt&logoColor=white">
  <img src="https://img.shields.io/badge/Socket-010101?style=for-the-badge&logo=socket.io&logoColor=white">
  <br>
  <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white">
  <img src="https://img.shields.io/badge/Confluence-172B4D?style=for-the-badge&logo=confluence&logoColor=white">
  <img src="https://img.shields.io/badge/Jira-0052CC?style=for-the-badge&logo=jira&logoColor=white">
</div>

<br>
<br>
<br>
<h2>Team Roles</h2>

| **팀원**       | **역할**                                        |
|-----------------------|--------------------------------------------------|
| 최희재 (팀장)         | 통합 서버 개발 (데이터베이스 관리, 통신)             |
| 김종호                | 딥러닝(포즈 인식, 객체 인식), 관리자 GUI             |
| 이헌중                | 딥러닝(라벨링, 객체 인식)                            |
| 조나온                | 딥러닝(얼굴인식), 통신, 사용자 GUI                   |


<br>
<br>
<h2>Deep Learning</h2>
<h3>포즈 인식</h3>

- 키포인트(사람의 뼈대)를 추출하는 세 가지 모델 검토 : YOLOv8-pose, Alpha pose, Media pipe
- 다중인물포즈 추정 이 가능하고 실시간처리 성능이 높으며 모델 복잡도가 다른 것들에 비하여 낮은 YOLOv8-pose를 채택

<img src="https://github.com/user-attachments/assets/996455eb-2b13-4cbe-8846-5598bc4b7dc9" width="720">
<img src="https://github.com/user-attachments/assets/3dfed97c-d238-4160-bf69-f2a9946ba37a" width="720">

<h3>얼굴 인식</h3>

- DeepFace 프레임워크를 사용하여 정확도, 속도를 고려한 모델 조합 구성
- 얼굴 감지 모델: RetinaFace, 얼굴 정렬: Enable, 인식 모델: FaceNet-512d, 거리 계산 : Cosine distance

<img src="https://github.com/user-attachments/assets/6c291903-57a2-4f8a-9ba5-1dd2ea5c5f44" width="720">
<img src="https://github.com/user-attachments/assets/ba7b8e59-fbbe-457e-8ea6-24647e10e4dd" width="720">

<br>
<br>
<h2>System Design</h2>
<h3>System Requirements</h3>

| **기능 분류**        | **세부 기능**                         | **기능 상세**                                                                 |
|-----------------------|---------------------------------------|-------------------------------------------------------------------------------|
| 얼굴 인식 결제        | 고객 식별 기능                        | 얼굴 인식기를 통해 매장에 방문한 고객의 신원을 식별하는 기능                  |
|                       | 장바구니 확인 기능                   | 고객이 장바구니에 담긴 과일 내역과 최종 금액을 확인하는 기능                  |
|                       | 결제 확인 기능                       | 고객이 장바구니 최종 금액을 확인하고 구매 의사를 확인하는 기능 (결제/취소)    |
|                       | 자동 결제 기능                       | 장바구니에 담긴 과일의 갯수만큼 결제된 금액을 사용자에게 UI로 전달하는 기능   |
| 도난 감지             | 매장 CCTV 실시간 모니터링 기능       | 관리자가 매장 상태를 실시간으로 확인할 수 있는 기능 (재고 정보, 고객 정보 포함)|
|                       | 녹화 기능                            | 시간대별로 녹화하여 보여주는 기능, 고객별 녹화 멀티 스레드 지원               |
|                       | 행위 내역 검색 기능                  | 특정 시간 및 행위 내역을 검색하고 확인 (예: 고객 진입, 퇴장, 물품 집음)        |
|                       | 도난 방지 및 알림 기능               | 도난 의심 행위를 감지 및 기록하고 알림을 제공하는 기능                         |
| 과일 상태 모니터링    | 물품 관리 기능                       | 매대 CCTV를 통해 정상/불량 상태 과일을 검출하고 총 수량 및 위치별 알림 기능 추가|
|                       | 실시간 매장 상태 모니터링 기능       | 관리자가 매장 상태를 확인하는 기능 (재고 정보, 고객 정보, CCTV 포함)          |
|                       | 판매 내역 검색 기능                  | 고객별, 물품별, 일/월별 조건에 따른 판매 내역을 검색하는 기능                 |

<br>
<h3>System Architecture</h3>
<img src="https://github.com/user-attachments/assets/1d6c5de1-490a-4ae6-89c9-f84b9e577df2" width="720">

<br>
<h3>Data Structure</h3>
<img src="https://github.com/user-attachments/assets/5111f65c-4ec5-4434-b79f-6b09d443b92c" width="720">

<br>
<h3>Scenarios</h3>

| 사람 입장 시나리오 | 장바구니 생성 시나리오 | 장바구니 관리 시나리오 |
|--------------------|----------------------|-----------------------|
| ![](https://github.com/user-attachments/assets/f827a47b-fb55-43c1-9eb9-c78d521fc827) | ![](https://github.com/user-attachments/assets/d1142336-1118-43b0-a612-a65278a13497) | ![](https://github.com/user-attachments/assets/b12b41f5-00ea-4400-b12e-b7a15a93e10c) |


<br>
<h3>GUI Design</h3>

- <h4>Admin GUI</h4>
<img src="https://github.com/user-attachments/assets/b979a699-bd2b-43f7-af24-c5c7c88ed9ec" width="480"> | <img src="https://github.com/user-attachments/assets/5fbf6b18-b53b-4315-a560-2e915e647656" width="480"> |

- <h4>Billing GUI</h4>
<img src="https://github.com/user-attachments/assets/28634150-2a06-4cca-8df1-527b12057db5" width="480"> | <img src="https://github.com/user-attachments/assets/5d9fc7c4-4994-4c60-83f0-a0f30040b44c" width="480"> |

<br>
<br>
<h2>Result</h2>

[![](https://github.com/user-attachments/assets/1645025f-8c75-4282-85f8-90ef94556128)](https://drive.google.com/file/d/1CZWh5j1DVTlnfuzvYG3wjAkVxbPhNY45/view?usp=drive_link)

<br>
<br>
<h2>Conclusion</h2>
<br>

딥러닝 기반의 무인 스마트 매장을 구현하였습니다.
- **RetinaFace, FaceNet-512**를 활용하여 얼굴 인식
- **YOLOv8-pose**를 활용하여 도난 행위 감지
- **YOLOv8**을 활용하여 과일 상태 분류
<br>



| **개선점**                         | **향후 계획**                                                                 |
|-------------------------------------|-------------------------------------------------------------------------------|
| **마스크, 모자 착용 시 인식 실패 문제**<br> 얼굴의 특징점(눈, 코, 입 등)이 가려지면 얼굴 인식 정확도가 저하됨. | **AI 모델 개선**<br> 노출된 얼굴 특징을 중심으로 인식률 높이는 방안 모색                                          |
| **도난 행위 오탐지 문제**<br> 사람의 관절 키포인트를 학습시키는 것으로는 사람의 행동의 디테일한 부분을 추적하는 데 한계가 있음.<br> 예: 고객이 스마트폰을 주머니에 넣는 과정에서 도난으로 감지되는 상황 | **AI 모델 개선**<br> 고객 행동의 다양성을 충분히 반영하는 데이터를 수집 및 학습하여 정밀도를 높이고 오탐지를 최소화 |

<br>
<br>
<br>
