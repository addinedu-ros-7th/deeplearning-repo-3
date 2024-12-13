import cv2
import numpy as np
import sys
import queue
import time

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot , QDate
from PyQt5.QtTest import QTest

from commons.logger import logger
from test_signin import CameraThread
#from test_TCPsender import TCPSenderThread
from test_TCPserver import TCPServerThread
from CartProcess import CartThread

logger = logger()

signinwindow = uic.loadUiType("mainwindow.ui")[0]
cartwindow = uic.loadUiType("cartwindow.ui")[0]
paymentwindow = uic.loadUiType("paymentwindow.ui")[0]


""" NOTE ------------------------------------------------

------------------------------------------------------"""


class SigninWindowClass(QMainWindow, signinwindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.member_queue = queue.Queue()
        self.cart_queue = queue.Queue()

        self.camera_thread = CameraThread(self.member_queue)
        self.camera_thread.update.connect(self.camera_update)
        self.camera_thread.signin_signal.connect(self.notice_signin)
        self.camera_thread.start()

        #self.tcp_sender_thread = TCPSenderThread(self.member_queue)
        self.tcpserver_thread = TCPServerThread(self.cart_queue)
        self.tcpserver_thread.start()
    
    def goto_next_window(self, cart):
        self.hide()
        self.next_window = CartWindowClass(cart)

    def sleep(self):
        QTest.qWait(1000)

    @pyqtSlot(np.ndarray)
    def camera_update(self, img):
            logger.info("frame in camera_update method")
            raw_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = raw_img.shape
            qimage = QImage(raw_img.data, w, h, w*c, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qimage)
            pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio)
            self.CameraLabel.setPixmap(pixmap)

    @pyqtSlot(int)
    def notice_signin(self, member_id):
        if member_id:
            logger.info("member_id in notice_signin method")

            self.sleep()
            logger.info("Waiting TCPServer")

            if self.cart_queue:
                try:
                    cart = self.cart_queue.get(timeout=1)
                    logger.info(f"cart_data in notice_signin : {cart}")
                    self.goto_next_window(cart)

                except Exception as e:
                    logger.info(f"Error in cart_queue : {e}")


class CartWindowClass(QWidget, cartwindow):
    def __init__(self, cart):
        super().__init__()
        self.setupUi(self)
        self.show()

        self.CartTable.setRowCount(0)
        self.CartTable.setColumnCount(3)
        self.CartTable.setHorizontalHeaderLabels(["Item", "Count", "Price"])

        self.MakePaymentButton.clicked.connect(self.goto_next_window)
        self.cart = cart
        if self.cart :
            self.listup_cart()

    def goto_next_window(self):
        self.hide()
        self.next_window = PaymentWindowClass()
    
    def listup_cart(self):
        if self.cart:
            logger.info("frame in camera_update method")
            items = self.cart['items']
            total_price = self.cart['total_price']

            self.CartTable.setRowCount(len(items))

            for i, item in enumerate(items):
                self.CartTable.setItem(i, 0, QTableWidgetItem(item["Item"]))
                self.CartTable.setItem(i, 1, QTableWidgetItem(str(item["Count"])))
                self.CartTable.setItem(i, 2, QTableWidgetItem(str(item["Price"])))
    
class PaymentWindowClass(QWidget, paymentwindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()

if __name__=='__main__':
    app = QApplication(sys.argv)
    signin_window = SigninWindowClass()
    signin_window.show()
    app.exec_()
