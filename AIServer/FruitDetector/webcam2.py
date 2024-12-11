import cv2

def test_webcam():
    # 웹캠 열기 (0은 기본 웹캠을 나타냅니다)
    cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("웹캠이 정상적으로 작동합니다. 'q' 키를 눌러 종료하세요.")

    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 프레임 표시
        cv2.imshow('Webcam Test2', frame)

        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

# 함수 실행
test_webcam()
