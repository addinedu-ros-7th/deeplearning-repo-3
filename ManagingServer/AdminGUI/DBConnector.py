from PyQt5.QtCore import QThread, pyqtSignal
import pymysql

class DBThread(QThread):
    # 각 테이블 데이터를 업데이트하는 신호 정의
    selling_log_signal = pyqtSignal(list)
    visit_log_signal = pyqtSignal(list)
    event_log_signal = pyqtSignal(list)
    selling_sum_signal = pyqtSignal(int)

    def __init__(self, db_config):
        super().__init__()
        self.db_config = db_config
        self.running = True
        self.date_filter = None

    def set_date_filter(self, date_filter):
        """날짜 필터 설정"""
        self.date_filter = date_filter

    def run(self):
        while self.running:
            try:
                # DB 연결
                conn = pymysql.connect(**self.db_config)
                cursor = conn.cursor()

                # 판매 로그 데이터 가져오기
                query = """
                    SELECT cart_fruit.cart_id, fruit.fruit_name, cart_fruit.quantity,
                           fruit.price * cart_fruit.quantity AS total_price, cart.pur_dttm
                    FROM cart_fruit
                    INNER JOIN fruit ON cart_fruit.fruit_id = fruit.fruit_id
                    INNER JOIN cart ON cart_fruit.cart_id = cart.cart_id
                """
                if self.date_filter:
                    query += f" WHERE DATE(cart.pur_dttm) = '{self.date_filter}'"

                cursor.execute(query)
                selling_log = list(cursor.fetchall())  # 튜플 리스트를 일반 리스트로 변환
                self.selling_log_signal.emit(selling_log)

                # 방문 로그 데이터 가져오기
                query = """
                    (
                        SELECT 
                            visit_info.visit_id, 
                            '입장' AS type, 
                            members.member_name, 
                            visit_info.in_dttm AS event_time
                        FROM visit_info
                        INNER JOIN members ON visit_info.member_id = members.member_id
                        WHERE visit_info.in_dttm IS NOT NULL
                    )
                    UNION ALL
                    (
                        SELECT 
                            visit_info.visit_id, 
                            '퇴장' AS type, 
                            members.member_name, 
                            visit_info.out_dttm AS event_time
                        FROM visit_info
                        INNER JOIN members ON visit_info.member_id = members.member_id
                        WHERE visit_info.out_dttm IS NOT NULL
                    )
                    ORDER BY event_time;
                """
                if self.date_filter:
                    query += f" WHERE DATE(event_time) = '{self.date_filter}'"
                #print(query)
                cursor.execute(query)
                visit_log = list(cursor.fetchall())  # 튜플 리스트를 일반 리스트로 변환
                #print("visit log : ",visit_log)
                self.visit_log_signal.emit(visit_log)

                # 이벤트 로그 데이터 가져오기
                query = """
                    SELECT event_info.event_id, '이벤트' AS type, members.member_name,
                           event_info.file_path, event_info.event_dttm
                    FROM event_info
                    INNER JOIN visit_info ON event_info.visit_id = visit_info.visit_id
                    INNER JOIN members ON visit_info.member_id = members.member_id
                """
                if self.date_filter:
                    query += f" WHERE DATE(event_info.event_dttm) = '{self.date_filter}'"

                
                cursor.execute(query)
                event_log = list(cursor.fetchall())  # 튜플 리스트를 일반 리스트로 변환
                self.event_log_signal.emit(event_log)

                conn.close()
            except pymysql.MySQLError as e:
                print(f"DB Error: {e}")
            except Exception as e:
                print(f"Unexpected Error: {e}")
            
            # 스레드 간 딜레이
            self.msleep(5000)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
