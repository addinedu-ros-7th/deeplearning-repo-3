class Cart:
    def __init__(self, camera_id, visit_id):
        self.camera_id = camera_id
        self.visit_id = visit_id
        self.cart_id = None
        self.fruits = {}



# Person 클래스를 정의 - 개별 사람의 상태를 추적하기 위해 사용
class Person:
    def __init__(self, track_id):
        self.track_id = track_id  # 각 사람의 고유 ID
        self.detected = False    # 사람이 감지되었는지 여부
        self.posed = False       # 포즈가 감지되었는지 여부