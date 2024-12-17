from PyQt5.QtCore import QObject
from PyQt5.QtNetwork import QTcpSocket
import threading
import json
import mysql.connector
from datetime import datetime
from custom_classes import *
from logger_config import setup_logger

logger = setup_logger()

class DataProcessor(QObject):
    def __init__(self):
        super().__init__()
        self.processors = {"Face": self.faceProcessor, 
                           "Fruit": self.fruitProcessor, 
                           "Cart": self.cartProcessor}
        self.lock = threading.Lock
        self.visitors = {}
        self.fruits = {}

        # cart 일단 1-4번 사용 가능으로 초기화
        self.available_carts = set(range(1,5))
        self.using_carts = set()

        # DB에 있는 과일 현황 업데이트
        self.fruits = {}
        conn = self.connectF2Mbase()
        cursor = conn.cursor()
        cursor.execute("select fruit_id, stock from fruit")
        for fruit_id, stock in cursor.fetchall():
            self.fruits[fruit_id] = stock

        cursor.execute("""
            SELECT  m.member_name, vi.member_id, vi.visit_id, c.cart_id, c.cart_cam, cf.fruit_id, cf.quantity, f.fruit_name, f.price
            FROM 
                members m
            LEFT JOIN 
                visit_info vi ON m.member_id = vi.member_id
            LEFT JOIN 
                cart c ON vi.visit_id = c.visit_id
            LEFT JOIN 
                cart_fruit cf ON c.cart_id = cf.cart_id
            LEFT JOIN 
                fruit f ON cf.fruit_id = f.fruit_id
            WHERE 
                vi.out_dttm IS NULL;
        """)

        # DB정보에 따라서 현재 매장 내 Visitor와 연결된 cart 업데이트
        data = cursor.fetchall()
        for row in data:
            member_name, member_id, visit_id, cart_id, cart_cam, fruit_id, quantity, fruit_name, price = row
            if visit_id not in self.visitors:
                if cart_id and cart_cam:
                    c = Cart(cart_id, cart_cam)
                    self.using_carts.add(cart_cam)
                else:
                    Cart(None, None)
                v = Visitor(visit_id, member_id, member_name, c)
                self.visitors[visit_id] = v
            if fruit_id:
                self.visitors[visit_id].cart.data[fruit_id] = [fruit_name, quantity, price]

        self.available_carts -= self.using_carts

        for v in self.visitors.values():
            logger.info(v)


    def faceProcessor(self, data):
        print(f"faceProcessor got {data}")
        data = data["data"][0]
        member_id = data["member_id"]
        action = data["action"]
        logger.info(f"member_id = {member_id}, action = {action}")

        # conn = self.connectF2Mbase()
        # cursor = conn.cursor()
        
        logger.info("looking for visitor")
        visitor = next(
            (v for v in self.visitors.values() if v.member_id == member_id),
            None,
        )

        if not visitor:
            logger.info("No visitor")
            cursor.execute("insert into visit_info (member_id, in_dttm) values (%s, %s)", (member_id, datetime.now()))
            conn.commit()

            cursor.execute("select max(visit_id) from visit_info where member_id=%s", (member_id,))
            result = cursor.fetchall()
            visit_id = result[0][0]

            cart_cam = self.assign_cart_cam()

            cursor.execute("insert into cart (visit_id, cart_cam, purchased) values (%s, %s, %s)", (visit_id, cart_cam, 0))
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
            logger.info(f"There's visitor: {visitor}")

            if visitor.cart.purchase == 0:
                print("entered purchase == 0")
                res = []
                # conn = self.connectF2Mbase()
                # cursor = conn.cursor()
                # # 해당 visitor의 cart에서 정보 가져오기.
                # cursor.execute("select f.fruit_name, f.price, cf.quantity \
                #                from cart_fruit cf \
                #                join fruit f on cf.fruit_id = f.fruit_id \
                #                where cf.cart_id=%s", (visitor.cart.cart_id,))
                
                # results = cursor.fetchall()
                # if results:
                #     logger.info(f"Visitor's cart_fruit results available")
                #     res_data = []
                #     for fruit_name, price, quantity in results:
                #         res_data.append({"Item": fruit_name, "Count": quantity, "Price": price})
                #     res["Items"] = res_data
                #     logger.info(res)
                # cursor.close()
                # conn.close()

                # if visitor.cart.data:
                #     for value in visitor.cart.data.values():
                #         res.append({"fruit_name": value[0], "count": value[1], "price": value[2]})


                # conn = self.connectF2Mbase()
                # cursor = conn.cursor()
                # cursor.execute("update cart set purchased=1 where cart_id=%s", (visitor.cart.cart_id,))
                # conn.commit()
                # cursor.close()
                # conn.close()

                #visitor.cart.purchase = 1
                # response queue에 put

                res.append({"fruit_name":"apple_fair", "count":10, "price": 1300})
                print(f"res = {res}")
                res = json.dumps(res).encode('utf-8')
                client_socket.write(res)
                print(f"send data to client: {client_socket.peerAddress().toString()}:{client_socket.peerPort()}")

            # 두번째 request의 경우
            elif visitor.cart.purchase == 1:
                if action == "yes":
                    conn = self.connectF2Mbase()
                    cursor = conn.cursor()
                    cursor.execute("update cart set purchased=%s where cart_id=%s", (2, visitor.cart.cart_id))
                    conn.close()
                    cursor.close()

                    # 계산을 끝낸 고객은 visitors에서 객체 삭제
                    logger.info(f"visitor {visitor.visit_id} deleted from visitors")
                    del self.visitors[visitor.visit_id]
                    self.res_data_queue.put(["submitted"])
                else:
                    # 아니오를 눌렀을 경우 계산하려고 얼굴 인식하기 전으로 visitor 객체 상태 초기화
                    visitor.cart.purchase = 0    

    def fruitProcessor(self, data):
        """
        data = [{"fruit_id":1, "stock": 3},
                {"fruit_id":2, "stock": 2},
                {"fruit_id":3, "stock": 2}]
        """
        data = data["data"]
        # with self.lock:
        conn = self.connectF2Mbase()

        cursor = conn.cursor()

        # ThreadManager의 fruit 딕셔너리 안에 저장되어 있는 이전 과일 매대 상태
        previous_fruit_ids = set(self.fruits.keys())
        logger.info(f"previous_fruit_ids: {previous_fruit_ids}")
        # 방금 받은 과일 매대 상태
        current_fruit_ids = set(map(int, [item["fruit_id"] for item in data]))
        logger.info(f"current_fruit_ids: {current_fruit_ids}")

        # 이전엔 있었는데 지금은 없는 과일 재고 0으로 설정
        zero_fruit_ids = previous_fruit_ids - current_fruit_ids
        logger.info(f"zero_fruit_ids: {zero_fruit_ids}")

        if zero_fruit_ids:
            query = "UPDATE fruit SET stock=0 WHERE fruit_id IN (%s)" % (
                ", ".join(["%s"] * len(zero_fruit_ids)))
            cursor.execute(query, tuple(zero_fruit_ids))
            logger.info(f"{zero_fruit_ids} updated to 0")


        # 과일 매대 재고 업데이트
        fruit_quantities = [(item["stock"], item["fruit_id"]) for item in data]
        logger.info(f"73: fruit_quantities: {fruit_quantities}")
        if fruit_quantities:
            query = "UPDATE fruit SET stock=%s where fruit_id=%s"
            cursor.executemany(query, fruit_quantities)
            logger.info(f"78: query executed")
        conn.commit()
        cursor.close()
        conn.close()

        # 업데이트 된 상태 다시 ThreadManager fruit 딕셔너리에 저장
        self.fruits = {item["fruit_id"]: item["stock"] for item in data}
        logger.info(f"85: self.fruits = {self.fruits}")



    def cartProcessor(self, data):
        """        
        data = [
                {"cart_cam": 1, "fruits": [{"2": 1, "4": 1}]}, 
                {"cart_cam": 2, "fruits": [{"2": 1, "4": 1}]}, 
                {"cart_cam": 3, "fruits": [{}]}, 
                {"cart_cam": 4, "fruits": [{}]}
                ]
        """
        using_carts = self.get_using_carts()
        logger.info(f"사용 중인 cart_cam: {using_carts}")
        #with self.lock:
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
                conn = self.connectF2Mbase()
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
                        cursor.execute("select fruit_name, price from fruit where fruit_id=%s", (fruit_id,))
                        results = cursor.fetchall()
                        visitor.cart.data[fruit_id] = [results[0][0], fruits[fruit_id], results[0][1]]
                        logger.info(f"insert into cart_fruit ({visitor.cart.cart_id, fruit_id, fruits[fruit_id]})")
                    conn.commit()

                    for fruit_id in common_fruit_ids:
                        cursor.execute("update cart_fruit set quantity=%s where fruit_id=%s and cart_id=%s", (fruits[fruit_id], fruit_id, visitor.cart.cart_id))
                        visitor.cart.data[fruit_id][1] = fruits[fruit_id]
                        logger.info(f"update cart_fruit {fruits[fruit_id]} where fruit_id={fruit_id} and cart_id={visitor.cart.cart_id} ")
                    conn.commit()

                    logger.info(
                        f"Cart Cam {cart_cam} 업데이트: {fruits}, "
                        f"Visitor ID: {visitor.visit_id}"
                        f"Visitor cart data: {visitor.cart.data}"
                    )
                else:
                    visitor.cart.data = {}
                    logger.info(f"visitor.cart.cart_id: {visitor.cart.cart_id}")
                    cursor.execute("delete from cart_fruit where cart_id=%s", (visitor.cart.cart_id,))
                    conn.commit()
                    logger.info(
                        f"Cart Cam {cart_cam} no fruits, "
                        f"Visitor ID: {visitor.visit_id} "
                        f"Visitor cart data: {visitor.cart.data}"
                    )

                cursor.close()
                conn.close()


    def connectF2Mbase(self):
        conn = mysql.connector.connect(
            host = "localhost",
            user = "root",
            password = "whdgh29k05",
            database="f2mdatabase"
        )
        return conn
        

    def assign_cart_cam(self):
        """사용 가능한 카트 할당"""
        #with self.lock:
        if not self.available_carts:
            return None  # 사용 가능한 카트가 없음
        cart_cam = self.available_carts.pop()
        self.using_carts.add(cart_cam)
        return cart_cam

    def release_cart_cam(self, cart_cam):
        """사용 중인 카트를 반환"""
        #with self.lock:
        if cart_cam in self.using_carts:
            self.using_carts.remove(cart_cam)
            self.available_carts.add(cart_cam)

    def get_using_carts(self):
        """현재 사용 중인 카트 반환"""
        #with self.lock:
        return list(self.using_carts)
