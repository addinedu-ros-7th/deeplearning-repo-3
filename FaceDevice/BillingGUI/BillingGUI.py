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

from sign_in import CameraThread
from client_socket import ClientThread
from server_socket import ServerThread

logger = logger()

signinwindow = uic.loadUiType("mainwindow.ui")[0]
cartwindow = uic.loadUiType("cartwindow.ui")[0]
paymentwindow = uic.loadUiType("paymentwindow.ui")[0]


class SigninWindowClass(QMainWindow, signinwindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("F2M.BillingGUI")
        self.setupUi(self)
        self.show()
        logger.info("open main window============================")
        self.member_queue = queue.Queue()
        self.cart_queue = queue.Queue()

        self.camera_thread = CameraThread()
        self.camera_thread.update.connect(self.camera_update)
        self.camera_thread.signin_signal.connect(self.notice_signin)

        self.client_thread = ClientThread(self.camera_thread)
        self.client_thread.start()

        self.server_thread = ServerThread()
        self.server_thread.cart_signal.connect(self.goto_next_window)
        self.server_thread.start()

    def closeEvent(self, event):
        if self.client_thread.isRunning():
            self.client_thread.stop()
        event.accept()
  
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
        if member:
            self.SigninTextLable.setText(f"Hello, {member}!")
            logger.info(f"Sign in : {member}")
            self.client_thread.send(member, False)
        else:
            self.SigninTextLable.setText("Unregistered user")

    @pyqtSlot(str)
    def goto_next_window(self, cart_str):
        self.hide()
        self.next_window = CartWindowClass(cart_str, self)
        #self.next_window.exec() # wait next window closing
        #self.show()             # re-show when next window is closed


class CartWindowClass(QWidget, cartwindow):
    def __init__(self, cart_str, signin_window):
        super().__init__()
        self.setupUi(self)
        self.show()
        logger.info("Open cart window ================")
        self.signin_window = signin_window

        self.CartTable.setRowCount(0)
        self.CartTable.setColumnCount(3)
        self.CartTable.setHorizontalHeaderLabels(["Item", "Count", "Price"])

        #self.GoBackButton.clicked.connect(self.goto_before_window)
        self.MakePaymentButton.clicked.connect(self.goto_next_window)

        self.cart = json.loads(cart_str)
        self.listup_cart()

    #def goto_before_window(self):
        
    def goto_next_window(self):
        self.hide()
        self.next_window = PaymentWindowClass(self.signin_window)

    def listup_cart(self):
        self.CartTable.setRowCount(len(self.cart))
        for i, item in enumerate(self.cart):
            self.CartTable.setItem(i, 0, QTableWidgetItem(item["fruit_name"]))
            self.CartTable.setItem(i, 1, QTableWidgetItem(str(item["count"])))
            self.CartTable.setItem(i, 2, QTableWidgetItem(str(item["price"])))


class PaymentWindowClass(QWidget, paymentwindow):
    def __init__(self, signin_window):
        super().__init__()
        self.setupUi(self)
        self.show()
        logger.info("open payment window=============")
        self.signin_window = signin_window
        QTest.qWait(1000)
        self.PaymentTextLable.setText("Payment Complete")
        QTest.qWait(1000)
        self.goto_main_window()

    def goto_main_window(self):
        self.hide()
        self.signin_window.show()


if __name__=='__main__':
    app = QApplication(sys.argv)
    signin_window = SigninWindowClass()
    signin_window.show()
    app.exec_()
