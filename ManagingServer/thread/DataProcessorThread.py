import threading
import queue
import json
import mysql.connector
from datetime import datetime
from custom_classes import *
from logger_config import setup_logger

logger = setup_logger()

class DataProcessorThread(threading.Thread):
    def __init__(self, camera_id, data_queue, res_data_queue, thread_manager):
        super().__init__()
        self.camera_id = camera_id
        self.data_queue = data_queue
        self.thread_manager = thread_manager
        self.visitors = self.thread_manager.visitors
        self.fruits = self.thread_manager.fruits
        self.lock = self.thread_manager.lock
        self._is_running = True

    def run(self):
        while self._is_running:
            try:
                data = self.data_queue.get(timeout=1)
                parsed_data = json.loads(data)
                
                if self.camera_id == "Face":
                    self.process_face_cam(parsed_data)
                elif self.camera_id == "Cart":
                    self.process_cart_cam(parsed_data)
                elif self.camera_id == "Fruit":
                    self.process_fruit_cam(parsed_data)

                self.data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"처리 중 오류: {str(e)}")

    def process_fruit_cam(self, data):
        # {1:3, 2:4, 5:7} {fruit_id: stock, fruit_id: stock...}
        data = self.thread_manager.convert_keys_to_int(data)

        with self.lock:
            conn = self.thread_manager.connect_f2mbase()

            cursor = conn.cursor()

            previous_fruit_ids = set(self.fruits.keys())
            print(previous_fruit_ids)
            current_fruit_ids = set(data.keys())
            print(current_fruit_ids)

            zero_fruit_ids = previous_fruit_ids - current_fruit_ids
            print(zero_fruit_ids)

            for fruit_id in zero_fruit_ids:
                cursor.execute("update fruit set stock=0 where fruit_id=%s", (fruit_id,))
            conn.commit()

            for fruit_id, stock in data.items():
                cursor.execute("update fruit set stock=%s where fruit_id=%s", (stock, fruit_id))
            conn.commit()
            cursor.close()
            conn.close()

            self.fruits = data



    def process_cart_cam(self, data):
        # data = {
        #     1: {1: 3, 2: 4},      cart_cam: {fruit_id: stock, fruit_id: stock}
        #     2: {1: 2, 5: 5},
        #     3: {4: 1},
        #     4: {3: 6}
        # }
        data = self.thread_manager.convert_keys_to_int(data)
        using_carts = self.thread_manager.get_using_carts()
        logger.info(f"사용 중인 cart_cam: {using_carts}")

        with self.lock:
            for cart_cam, data in data.items():
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

                    previous_fruit_ids = set(visitor.cart.data.keys())
                    current_fruit_ids = set(data.keys())

                    zero_fruit_ids = previous_fruit_ids - current_fruit_ids

                    new_fruit_ids = current_fruit_ids - previous_fruit_ids

                    common_fruit_ids = previous_fruit_ids & current_fruit_ids

                    for fruit_id in zero_fruit_ids:
                        cursor.execute("delete from cart_fruit where cart_id=%s and fruit_id=%s", (visitor.cart.cart_id, fruit_id),)
                    conn.commit()

                    for fruit_id in new_fruit_ids:
                        cursor.execute("insert into cart_fruit (cart_id, fruit_id, quantity) values (%s, %s, %s)", (visitor.cart.cart_id, fruit_id, data[fruit_id]))
                    conn.commit()

                    for fruit_id in common_fruit_ids:
                        cursor.execute("update fruit set stock=%s where fruit_id=%s", (data[fruit_id], fruit_id))
                    conn.commit()

                    cursor.close()
                    conn.close()
                    
                    visitor.cart.update(data)

                    logger.info(
                        f"Cart Cam {cart_cam} 업데이트: {data}, "
                        f"Visitor ID: {visitor.visit_id}"
                    )

    def process_purchase_cam(self, data):
        member_id = data["member_id"]

        visitor = next(
            (v for v in self.visitors.values() if v.member_id == member_id),
            None,
        )

        if visitor:
            request_id = data["request_id"]
            
            if request_id == 1 and visitor.cart.purchase == 0:
                conn = self.thread_manager.connect_f2mbase()
                cursor = conn.cursor()





    def process_face_cam(self, data):
        member_id = data["member_id"]

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
        else:
            # db에 out_dttm이랑 cart정보, purchased 정보 올리고 객체 소멸, 카트 unassign
            cursor.execute("update visit_info set out_dttm=%s where visit_id=%s", (datetime.now(), visitor.visit_id))
            conn.commit()


            self.thread_manager.release_cart_cam(visitor.cart.cart_cam)
            del self.visitors[visitor.visit_id]

            logger.info(f"Visitor 삭제: {member_id} -> Visit ID {visitor.visit_id}")
        
        cursor.close()
        conn.close()

    def stop(self):
        self._is_running = False
