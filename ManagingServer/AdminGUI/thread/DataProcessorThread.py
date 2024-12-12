import threading
import queue
import json
import mysql.connector
from datetime import datetime
from .custom_classes import *
from .logger_config import setup_logger

logger = setup_logger()

class DataProcessorThread(threading.Thread):
    def __init__(self, data_queue, res_data_queue, thread_manager):
        super().__init__()
        #self.camera_id = camera_id
        self.data_queue = data_queue
        self.res_data_queue = res_data_queue
        self.thread_manager = thread_manager
        self.visitors = self.thread_manager.visitors
        self.fruits = self.thread_manager.fruits
        self.lock = self.thread_manager.lock
        self._is_running = True

    def run(self):
        while self._is_running:
            try:
                data = self.data_queue.get(timeout=1)[1]
                parsed_data = json.loads(data)
                
                print(parsed_data)
                if parsed_data["camera_id"] == "Face":
                    self.process_face_cam(parsed_data["data"])
                elif parsed_data["camera_id"] == "Cart":
                    self.process_cart_cam(parsed_data["data"])
                elif parsed_data["camera_id"] == "Fruit":
                    self.process_fruit_cam(parsed_data["data"])

                self.data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"처리 중 오류: {str(e)}")

    def process_fruit_cam(self, data):
        with self.lock:
            conn = self.thread_manager.connect_f2mbase()

            cursor = conn.cursor()

            # ThreadManager의 fruit 딕셔너리 안에 저장되어 있는 이전 과일 매대 상태
            previous_fruit_ids = set(self.fruits.keys())
            # 방금 받은 과일 매대 상태
            current_fruit_ids = set(map(int, [item["fruit_id"] for item in data]))

            # 이전엔 있었는데 지금은 없는 과일 재고 0으로 설정
            zero_fruit_ids = previous_fruit_ids - current_fruit_ids

            if zero_fruit_ids:
                query = "UPDATE fruit SET stock=0 WHERE fruit_id IN (%s)" % (
                    ", ".join(["%s"] * len(zero_fruit_ids)))
                cursor.execute(query, tuple(zero_fruit_ids))

            # 과일 매대 재고 업데이트
            fruit_quantities = [(item["quantity"], item["fruit_id"]) for item in data]
            if fruit_quantities:
                query = "INSERT INTO fruit (stock, fruit_id) VALUES (%s, %s) " \
                        "ON DUPLICATE KEY UPDATE stock=VALUES(stock)"
                cursor.executemany(query, fruit_quantities)
            conn.commit()
            cursor.close()
            conn.close()

            # 업데이트 된 상태 다시 ThreadManager fruit 딕셔너리에 저장
            self.fruits = {item["fruit_id"]: item["quantity"] for item in data}



    def process_cart_cam(self, data):
        """        
        data = [{"cart_cam": 1,
                 "fruits": [ 
                    {1: 3, 2: 4, 3: 5}
                 ]},
                 {"cart_cam": 2,
                  "fruits": [
                    {1: 2, 2: 3, 3: 5}
                 ]}
                 ]
        """
        using_carts = self.thread_manager.get_using_carts()
        logger.info(f"사용 중인 cart_cam: {using_carts}")
        with self.lock:
            for cart in data:
                # cart = {"cart_cam": 1, "fruits": [{1: 3, 2: 4, 3: 5}]}
                cart_cam = cart["cart_cam"]
                logger.info(f"data's cart_cam: {cart_cam}")
                # fruits = {1: 3, 2: 4, 3: 5}
                # empty 라면 {}
                fruits = cart["fruits"][0] # 리스트에서 dict 꺼냄
                logger.info(f"data's fruits: {fruits}")
                if cart_cam in using_carts:  # 사용 중인 카트만 처리
                    cart_cam = cart_cam
                    # 해당 cart_cam을 사용하는 방문자를 찾음
                    visitor = next(
                        (v for v in self.visitors.values() if v.cart.cart_cam == cart_cam),
                        None,
                    )

                    if not visitor:
                        logger.warning(f"Visitor not found for cart_cam {cart_cam}")
                        continue

                    
                    logger.info(f"visit_id: {visitor.visit_id}, cart_cam: {visitor.cart.cart_cam}")
                    conn = self.thread_manager.connect_f2mbase()
                    cursor = conn.cursor()

                    if fruits:
                        current_fruit_ids = set(fruits.keys())
                        previous_fruit_ids = set(visitor.cart.data.keys())
                        

                        zero_fruit_ids = previous_fruit_ids - current_fruit_ids
                        logger.info(f"zero_fruit_ids: {zero_fruit_ids}")

                        new_fruit_ids = current_fruit_ids - previous_fruit_ids
                        logger.info(f"new_fruit_ids: {new_fruit_ids}")

                        common_fruit_ids = previous_fruit_ids & current_fruit_ids
                        logger.info(f"common_fruit_ids: {common_fruit_ids}")

                        for fruit_id in zero_fruit_ids:
                            cursor.execute("delete from cart_fruit where cart_id=%s and fruit_id=%s", (visitor.cart.cart_id, fruit_id))
                            logger.info(f"delete from cart_fruit ({visitor.cart.cart_id, fruit_id})")
                        conn.commit()

                        for fruit_id in new_fruit_ids:
                            cursor.execute("insert into cart_fruit (cart_id, fruit_id, quantity) values (%s, %s, %s)", (visitor.cart.cart_id, fruit_id, fruits[fruit_id]))
                            logger.info(f"insert into cart_fruit ({visitor.cart.cart_id, fruit_id, fruits[fruit_id]})")
                        conn.commit()

                        for fruit_id in common_fruit_ids:
                            cursor.execute("update fruit set stock=%s where fruit_id=%s", (fruits[fruit_id], fruit_id))
                            logger.info(f"update cart_fruit ({fruits[fruit_id], fruit_id})")
                        conn.commit()
                        
                        visitor.cart.update(fruits)

                        logger.info(
                            f"Cart Cam {cart_cam} 업데이트: {fruits}, "
                            f"Visitor ID: {visitor.visit_id}"
                            f"Visitor cart data: {visitor.cart.data}"
                        )
                    else:
                        visitor.cart.update({})
                        cursor.execute("delete from cart_fruit where cart_id=%s", (visitor.cart.cart_id))
                        logger.info(
                            f"Cart Cam {cart_cam} no fruits, "
                            f"Visitor ID: {visitor.visit_id}"
                            f"Visitor cart data: {visitor.cart.data}"
                        )

                    cursor.close()
                    conn.close()


    def process_face_cam(self, data):
        data = data[0]
        print(f"process_face_cam accepted data {data}")
        member_id = data["member_id"]
        action = data["action"]

        conn = self.thread_manager.connect_f2mbase()
        cursor = conn.cursor()

        visitor = next(
            (v for v in self.visitors.values() if v.member_id == member_id),
            None,
        )

        if not visitor:
            cursor.execute("insert into visit_info (member_id, in_dttm) values (%s, %s)", (member_id, datetime.now()))
            conn.commit()

            cursor.execute("select max(visit_id) from visit_info where member_id=%s", (member_id,))
            result = cursor.fetchall()
            visit_id = result[0][0]

            cart_cam = self.thread_manager.assign_cart_cam()

            cursor.execute("insert into cart (visit_id, cart_cam) values (%s, %s)", (visit_id, cart_cam))
            conn.commit()

            cursor.execute("select max(cart_id) from cart where visit_id=%s", (visit_id,))
            result = cursor.fetchall()
            cart_id = result[0][0]

            c = Cart(cart_id=cart_id, cart_cam=cart_cam)
            v = Visitor(visit_id, member_id, c)
            logger.info(f"Visitor 객체 생성: {member_id}, {visit_id}, {cart_id}, {cart_cam}")

            with self.lock:
                self.visitors[visit_id] = v

            logger.info(f"Visitor 추가: {member_id} -> Visit ID {visit_id}")
            cursor.close()
            conn.close()
        else:
            # 에러
            if visitor.cart.purchase == 0:
                res = {}
                conn = self.thread_manager.connect_f2mbase()
                cursor = conn.cursor()
                # 해당 visitor의 cart에서 정보 가져오기.
                cursor.execute("select f.fruit_name, f.price, cf.quantity \
                               from cart_fruit cf \
                               join fruit f on cf.fruit_id = f.fruit_id \
                               where cf.cart_id=%s", (visitor.cart.cart_id))
                
                results = cursor.fetchall()
                if results:
                    res_data = []
                    for fruit_name, price, quantity in results:
                        res_data.append({"Item": fruit_name, "Count": quantity, "Price": price})
                    res["Items"] = res_data
                cursor.close()
                conn.close()

                visitor.cart.purchase = 1
                # response queue에 put

                self.res_data_queue.put(res)

            # 두번째 request의 경우
            elif visitor.cart.purchase == 1:
                if action == "yes":
                    conn = self.thread_manager.connect_f2mbase()
                    cursor = conn.cursor()
                    cursor.execute("update cart set purchase=%s where cart_id=%s", (2, visitor.cart.cart_id))
                    conn.close()
                    cursor.close()

                    # 계산을 끝낸 고객은 visitors에서 객체 삭제
                    logger.info(f"visitor {visitor.visit_id} deleted from visitors")
                    del self.visitors[visitor.visit_id]
                    self.res_data_queue.put(["submitted"])
                else:
                    # 아니오를 눌렀을 경우 계산하려고 얼굴 인식하기 전으로 visitor 객체 상태 초기화
                    visitor.cart.purchase = 0
        


    def stop(self):
        self._is_running = False
