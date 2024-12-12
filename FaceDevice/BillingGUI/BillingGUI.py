import sys
import cv2
import numpy as np

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot , QDate

from commons.logger import logger
#from commons.database import database

from SignInProcess import CameraThread, TCPSender
#from CheckCartProcess import ListupCart

logger = logger()

signinwindow = uic.loadUiType("mainwindow.ui")[0]
cartwindow = uic.loadUiType("cartwindow.ui")[0]
paymentwindow = uic.loadUiType("paymentwindow.ui")[0]


class SigninWindowClass(QMainWindow, signinwindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("F2M.BillingGUI")
        self.setupUi(self)

        self.camera_thread = CameraThread()
        self.camera_thread.update.connect(self.camera_update)
        self.camera_thread.signin_signal.connect(self.notice_signin)
        self.camera_thread.start()
    """
    def goto_next_window(self):
        self.hide()
        self.next_window = CartWindowClass()
        #self.next_window.exec() # wait next window closing
        #self.show()             # re-show when next window is closed
    """
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
        print(f"received : {member}")
        if member is not None:
            self.SigninTextLable.setText(f"Hello, {member}!")
            self.camera_thread.stop()
            #self.goto_next_window()
        else:
            self.SigninTextLable.setText("Unregistered user")

"""
class CartWindowClass(QWidget, cartwindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()

        #self.cart_thread.cart_signal.connect(self.notice_payment)
        #self.MakePaymentButton.clicked.connect(self.make_payment)

    # @pyqtSlot()
    # def cart_list(self):
        # AdminGUI로부터 member의 cart 데이터를 수신하여 디스플레이
        # received data type : json 
        # ex: { fruit1: 3, fruit2 : 4, fruit3 :  5} 

    
    def notice_payment(self):
        self.PaymentTextBox.setText("Please make payment")


    def make_payment(self):
        self.ResultTextBox.setText("Success!")

"""

if __name__=='__main__':
    app = QApplication(sys.argv)
    signin_window = SigninWindowClass()
    signin_window.show()
    app.exec_()

