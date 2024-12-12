import sys
import cv2
import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot , QDate
import time
from commons.logger import logger
#from commons.database import database

from SignInProcess import CameraThread
from CartProcess import CartThread
from TCPSenderThread import TCPSenderThread
from TCPServerThread import TCPServerThread
import queue

logger = logger()

signinwindow = uic.loadUiType("mainwindow.ui")[0]
cartwindow = uic.loadUiType("cartwindow.ui")[0]
paymentwindow = uic.loadUiType("paymentwindow.ui")[0]


class SigninWindowClass(QMainWindow, signinwindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("F2M.BillingGUI")
        self.setupUi(self)
        self.member_queue = queue.Queue()
        self.cart_queue = queue.Queue()

        self.camera_thread = CameraThread(self.member_queue)
        self.camera_thread.update.connect(self.camera_update)
        self.camera_thread.signin_signal.connect(self.notice_signin)
        self.camera_thread.start()

        self.tcp_sender_thread = TCPSenderThread(self.member_queue)
        self.tcp_sender_thread.run()
        
        self.tcp_server_thread = TCPServerThread(self.cart_queue)
        self.tcp_server_thread.run()

    def goto_next_window(self):
        self.hide()
        self.next_window = CartWindowClass(self.cart_queue)
        #self.next_window.exec() # wait next window closing
        #self.show()             # re-show when next window is closed

    @pyqtSlot(np.ndarray)
    def camera_update(self, img):
        raw_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = raw_img.shape
        qimage = QImage(raw_img.data, w, h, w*c, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio)
        self.CameraLabel.setPixmap(pixmap)

    @pyqtSlot(int)
    def notice_signin(self, member):
        logger.info(f"received : {member}")
        if member is not None:
            self.SigninTextLable.setText(f"Hello, {member}!")
            self.camera_thread.stop()
            time.sleep(50)
            try:
                cart = self.cart_queue.get(timeout=3)
                print(cart)
                self.goto_next_window()
            except Exception as e :
                logger.info(f"Cart Queue from Server is None: {e}")
            
        else:
            self.SigninTextLable.setText("Unregistered user")


class CartWindowClass(QWidget, cartwindow):
    def __init__(self, cart_queue):
        super().__init__()
        self.setupUi(self)
        self.show()
        self.cart_queue = cart_queue

        self.CartTable.setRowCount(0)
        self.CartTable.setColumnCount(3)
        self.CartTable.setHorizontalHeaderLabels(["Item", "Count", "Price"])

        self.cart_thread = CartThread(self.cart_queue)
        self.cart_thread.cart_signal.connect(self.listup_cart)
        self.cart_thread.run()
        #self.GoBackButton.clicked.connect(self.goto_before_window)
        self.MakePaymentButton.clicked.connect(self.goto_next_window)

    #def goto_before_window(self):
        
    def goto_next_window(self):
        self.hide()
        self.next_window = PaymentWindowClass()

    @pyqtSlot(dict)
    def listup_cart(self, cart):
        print(type(cart))
        items = cart["items"]
        total_price = cart["total_price"]
        print(total_price)
        self.cart_thread.stop()

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
