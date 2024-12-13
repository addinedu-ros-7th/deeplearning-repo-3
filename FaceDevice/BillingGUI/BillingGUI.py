import sys
import cv2
import numpy as np
import queue
import json

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot , QDate
from PyQt5.QtTest import QTest

from commons.logger import logger
#from commons.database import database

from SignInProcess import CameraThread
from CartProcess import CartThread
from TCPSenderThread import TCPSenderThread
from TCPServerThread import TCPServerThread

logger = logger()

signinwindow = uic.loadUiType("mainwindow.ui")[0]
cartwindow = uic.loadUiType("cartwindow.ui")[0]
paymentwindow = uic.loadUiType("paymentwindow.ui")[0]

def sleep():
    QTest.qWait(3000)

class SigninWindowClass(QMainWindow, signinwindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("F2M.BillingGUI")
        self.setupUi(self)
        self.show()

        self.member_queue = queue.Queue()
        self.cart_queue = queue.Queue()

        self.camera_thread = CameraThread(self.member_queue)
        self.camera_thread.update.connect(self.camera_update)
        self.camera_thread.signin_signal.connect(self.notice_signin)
        self.camera_thread.start()

        self.tcp_sender_thread = TCPSenderThread(self.member_queue)
        self.tcp_sender_thread.start()
        
        self.tcp_server_thread = TCPServerThread(self.cart_queue)
        self.tcp_server_thread.start()

        self.cart_thread = CartThread(self.cart_queue)
        self.cart_thread.cart_signal.connect(self.goto_next_window)
        self.cart_thread.start()

        
    @pyqtSlot(str)
    def goto_next_window(self, cart_str):
        logger.info(f"cart_data in goto_next_window : {cart_str}")
        #self.member_queue.put(cart_str)
        
        self.hide()
        self.next_window = CartWindowClass(cart_str)
        #self.next_window.exec() # wait next window closing
        #self.show()             # re-show when next window is closed

    @pyqtSlot(np.ndarray)
    def camera_update(self, img):
        #logger.info("frame in camera_update method")
        raw_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = raw_img.shape
        qimage = QImage(raw_img.data, w, h, w*c, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio)
        self.CameraLabel.setPixmap(pixmap)

    @pyqtSlot(int)
    def notice_signin(self, member):
        if member:
            logger.info(f"member_id in notice_signin method : {member}")
            self.SigninTextLable.setText(f"Hello, {member}!")
        else:
            self.SigninTextLable.setText("Unregistered user")

class CartWindowClass(QWidget, cartwindow):
    def __init__(self, cart_str):
        super().__init__()
        self.setupUi(self)
        self.show()

        self.CartTable.setRowCount(0)
        self.CartTable.setColumnCount(3)
        self.CartTable.setHorizontalHeaderLabels(["Item", "Count", "Price"])

        self.MakePaymentButton.clicked.connect(self.goto_next_window)

        #self.GoBackButton.clicked.connect(self.goto_before_window)
        self.MakePaymentButton.clicked.connect(self.goto_next_window)

        self.cart = json.loads(cart_str)
        self.listup_cart()

    #def goto_before_window(self):
        
    def goto_next_window(self):
        self.hide()
        self.next_window = PaymentWindowClass()

    def listup_cart(self):
        print(f"listcart has data : {self.cart}")
        #items = self.cart['items']
        #total_price = self.cart['total_price']

        self.CartTable.setRowCount(len(self.cart))
        for i, item in enumerate(self.cart):
            self.CartTable.setItem(i, 0, QTableWidgetItem(item["fruit_name"]))
            self.CartTable.setItem(i, 1, QTableWidgetItem(str(item["count"])))
            self.CartTable.setItem(i, 2, QTableWidgetItem(str(item["price"])))
    
class PaymentWindowClass(QWidget, paymentwindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()

        self.goto_main_window()

    def goto_main_window(self):
        sleep()
        self.hide()
        self.next_window = SigninWindowClass()
        self.next_window.show()

if __name__=='__main__':
    app = QApplication(sys.argv)
    signin_window = SigninWindowClass()
    signin_window.show()
    app.exec_()
