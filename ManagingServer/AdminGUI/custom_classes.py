class Cart:
    def __init__(self, cart_id, cart_cam):
        self.cart_id = cart_id
        self.cart_cam = cart_cam
        self.data = {}  # {fruit_id: [name, stock, price], fruit_id: [name, stock, price]...}
        self.purchase = 0

    def update(self, new_data):
        self.data = new_data

    def get_data(self):
        return self.data.copy()
    
    def __str__(self):
        return (
            f"Cart(cart_id={self.cart_id}, cart_cam={self.cart_cam}, "
            f"data={self.data}), "
            f"purchase={self.purchase}"
        )

    def __repr__(self):
        return self.__str__()

class Event:
    def __init__(self, event_id):
        self.event_id = event_id
        self.data = {}

class Visitor:
    def __init__(self, visit_id, member_id, member_name, cart):
        self.visit_id = visit_id
        self.member_id = member_id
        self.member_name = member_name
        self.cart = cart

    def __str__(self):
        return (
            f"Visitor(visit_id={self.visit_id}, member_id={self.member_id}, "
            f"cart={self.cart})"
        )

    def __repr__(self):
        return self.__str__()


# Person 클래스를 정의 - 개별 사람의 상태를 추적하기 위해 사용
class Person:
    def __init__(self, track_id, visit_id):
        self.visit_id = visit_id
        self.track_id = track_id  # 각 사람의 고유 ID
        self.detected = False    # 사람이 감지되었는지 여부
        self.posed = False       # 포즈가 감지되었는지 여부