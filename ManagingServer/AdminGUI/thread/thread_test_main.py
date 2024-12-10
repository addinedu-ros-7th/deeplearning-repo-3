if __name__ == "__main__":
    import time
    from ThreadManager import ThreadManager


    # 메인 스레드에서 CameraManager 생성
    thread_manager = ThreadManager()

    try:
        # 카메라 추가
        thread_manager.add_camera(camera_id="Face", client="0.0.0.0", port=5001)
        thread_manager.add_camera(camera_id="Cart", client="0.0.0.0", port=5002)
        thread_manager.add_camera(camera_id="Fruit", client="0.0.0.0", port=5003)


    except KeyboardInterrupt:
        print("프로그램 종료 중...")
        thread_manager.stop_all()


    
