import threading
import cv2
import numpy as np

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from commons.logger import logger
from commons import dir_utils
from modules.modeling import modeling
from recognition_class import recognition


logger = logger()

class CameraThread(QThread):
    update = pyqtSignal(np.ndarray)
    signin_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super(CameraThread, self).__init__(parent)
        self.recognition = recognition(self.signin_signal)
        self.db_path = dir_utils.initialize_dir()
        self.models = modeling["models"]
        self.detector_backend=modeling["backends"]
        self.distance_metric=modeling["metrics"]
        self.running = True

    def run(self):
        logger.info("CameraThread is starting")
        cap = cv2.VideoCapture(0)

        while self.running:
            #try:
            #    result = self.recognition.analysis(db_path=self.db_path,
            #                             model_name="Facenet",
            #                             detector_backend=modeling["backends"][4],
            #                             distance_metric=modeling["metrics"][1],
            #                             time_threshold=3,
            #                             )
            #    self.stop()
            #except Exception as e:
            #    logger.error(f"Error in recognition : {e}")

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
