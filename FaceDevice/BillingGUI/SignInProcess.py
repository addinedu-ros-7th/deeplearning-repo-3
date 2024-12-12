from deepface import DeepFace
import realtime_face_recognition

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

from commons.logger import logger

import threading
import queue
import time
import socket
import json
import cv2
import sys
import os
import glob
import numpy as np

#-------------------Variable Setting-------------------
# Input data source : "camera", "video", "image"
DATA_SOURCE = "camera"

# Camera(Webcam)
CAM_NUM = 0

# Image directory path
IMAGE_DIRECTORY_PATH = "data/face_samples/"
#DATABASE_DRECTORY_PATH = "data/face_database/"
DATABASE_DRECTORY_PATH = "customer_database/"
VIDEO_DIRECTORY_PATH = "data/video/"

SERVER_IP = '192.168.0.100'
SERVER_PORT = 5001
CAMERA_ID = 2
# -----------------------------------------------------


logger = logger()
member_queue = queue.Queue()

class CameraThread(QThread):
    update = pyqtSignal(np.ndarray)
    signin_signal = pyqtSignal(int)

    def __init__(self, db_path=DATABASE_DRECTORY_PATH, member_queue=member_queue):
        super().__init__()
        self.models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
        self.backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
        self.metrics = ["cosine", "euclidean", "euclidean_l2"]
        self.db_path = db_path
        self.member_queue = member_queue
        self.running = True
        
    def run(self):
        logger.info("Starting camera")
        cap = cv2.VideoCapture(CAM_NUM)

        while True:
            has_frame, frame = cap.read()
            if not has_frame:
                logger.warning("Failed to capture frame from the camera. Check camera connection or configuration.")
                break
            self.update.emit(frame)

            # Perform face recognization with DeepFace Module before running below lines---------------------------
            member_id=11    # member id for test
            if member_id:
                logger.info(f"member id : {member_id}")
                try:
                    self.signin_signal.emit(member_id)

                except Exception as e:
                    logger.error("Error in Signin thread: %s", e)
                    break

            #self.member_queue.put(member_id)
            #logger.info(f"Current queue: {list(member_queue.queue)}")
            #-------------------------------------------------------------------------------------------------------

            if not self.running:
                break
            
        cap.release()

    def stop(self):
        self.running = False
        logger.info("Signin thread stopping")


class TCPSender(QThread):

    def __init__(self, server_ip, server_port, camera_id, member_queue=member_queue):
        super().__init__()
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_ip, self.server_port))
        self.camera_id = camera_id
        logger.info(f"Connected to {self.server_ip}:{self.server_port}")

        self.member_queue = member_queue
        self.running = True
    
    def run(self):
        logger.info("TCPSender thread started: %s", threading.currentThread().getName())
        while self.running: 
            try:
                name = self.member_queue.get(timeout=1)
                logger.info("Received from queue: %s", name)

                data = {"camera_id": self.camera_id, "member_id": name}
                self.client_socket.send(json.dumps(data).encode())
                logger.info(f"Data is sent : {data}")

            except queue.Empty:
                logger.warning("Queue is empty.")
                continue

            except (BrokenPipeError, socket.error) as e:
                logger.error("Error in TCPSender thread: %s", e)
                break

    def stop(self):
        self.running = False
        logger.info("TCPSender thread stopping")


