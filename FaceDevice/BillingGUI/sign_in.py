import threading
import cv2
import numpy as np

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from commons.logger import logger
from commons import dir_utils


logger = logger()

class CameraThread(QThread):
    update = pyqtSignal(np.ndarray)
    signin_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super(CameraThread, self).__init__(parent)
        self.running = True

    def run(self):
        logger.info("CameraThread is starting")
        cap = cv2.VideoCapture(0)

        while self.running:
            try:
                has_frame, frame = cap.read()
                if not has_frame:
                    logger.warning("Failed to capture frame from the camera. Check camera connection or configuration.")
                    break
                self.update.emit(frame)

                # Perform face recognization with DeepFace Module before running below lines---------------------------
                member_id = 1
                if member_id:
                    logger.info(f"Face Detected : {member_id}")
                    self.signin_signal.emit(member_id)
                    break
                #-------------------------------------------------------------------------------------------------------
            
            except Exception as e:
                logger.error("Error in CameraThread running: %s", e)
                self.running = False 

    def stop(self):
        self.running = False
        logger.info("CameraThread has been stopped")
