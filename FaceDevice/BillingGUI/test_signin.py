import cv2
import numpy as np
import sys
import queue

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot , QDate

from commons.logger import logger
#from commons.database import database

CAM_NUM = 0

logger = logger()

class CameraThread(QThread):
    update = pyqtSignal(np.ndarray)
    signin_signal = pyqtSignal(int)

    def __init__(self, member_qeuue):
        super().__init__()
        self.member_queue = member_qeuue
        self.running = True

        logger.info("CameraThread starting")
    
    def run(self):
        cap = cv2.VideoCapture(CAM_NUM)
        while self.running:
            try:
                has_frame, frame = cap.read()
                if not has_frame:
                    logger.warning("Failed to capture frame from the camera. Check camera connection or configuration.")
                    break
                self.update.emit(frame)
                logger.info(f"Emit frame as update signal : {type(frame)}")

                member_id = 1
                if member_id:
                    logger.info(f"Face Detected : {member_id}")
                    self.member_queue.put(member_id)
                    logger.info(f"Put member_id to queue: {member_id}")
                    self.signin_signal.emit(member_id)
                    logger.info(f"Emit member_id as signin signal : {member_id}")
                    break
                
            except Exception as e:
                logger.error("Error in CameraThread running: %s", e)
                self.running = False


    def stop(self):
        self.running = False
        logger.info("CameraThread stopping")    
