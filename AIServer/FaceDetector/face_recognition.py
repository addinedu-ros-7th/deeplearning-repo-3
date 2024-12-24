from deepface import DeepFace
import realtime_face_recognition
import logging
import threading
import queue
import time
import socket
import json
import cv2
import sys
import os
import glob

from commons import dir_utils
from commons import socket_setups
"""NOTE-------------------------------------------------
- DeepFace.find(
            ...
            threshold=20
            ...)
 ERROR - Error in FaceRecognition thread: not enough values to unpack (expected 3, got 2)
-----------------------------------------------------"""

#-------------------Variable Setting-------------------
# Input data source : "camera", "video", "image"
DATA_SOURCE = "camera"

# Camera(Webcam)
CAM_NUM = 0

# Image directory path
IMAGE_DIRECTORY_PATH = "data/face_samples/"
DATABASE_DRECTORY_PATH = dir_utils.initialize_dir()
VIDEO_DIRECTORY_PATH = "data/video/"
# -----------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    #level=logging.INFO,q
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FaceRecognition(threading.Thread):
    def __init__(self, db_path, name_queue):
        super().__init__()
        self.models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
        self.backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
        self.metrics = ["cosine", "euclidean", "euclidean_l2"]
        self.db_path = db_path
        self.name_queue = name_queue
        self.running = True

    
    def run(self):
        logger.info("FaceRecognition thread started: %s", threading.currentThread().getName())
        try:
            result = realtime_face_recognition.analysis(db_path=self.db_path,
                                     model_name=self.models[2],
                                     detector_backend=self.backends[4],
                                     distance_metric=self.metrics[1],
                                     time_threshold=3,
                                     name_queue=self.name_queue
                                     )
            self.stop()
        except Exception as e:
            logger.error("Error in FaceRecognition thread: %s", e)
        

    def stop(self):
        self.running = False
        logger.info("FaceRecognition thread stopping")


class TCPSender(threading.Thread):
    def __init__(self, server_ip, server_port, camera_id, name_queue):
        super().__init__()
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_ip, self.server_port))
        self.camera_id = camera_id
        logger.info(f"Connected to {self.server_ip}:{self.server_port}")

        self.name_queue = name_queue
        self.running = True
    
    def run(self):
        logger.info("TCPSender thread started: %s", threading.currentThread().getName())
        while self.running: 
            try:
                name = self.name_queue.get(timeout=1)
                logger.info("Received from queue: %s", name)

                data = {"camera_id": self.camera_id, "data": [{"member_id": name,  "action": "visit"}]}
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


def main():
    logger.info("Application starting")
    name_queue = queue.Queue()

    recognition_thread = FaceRecognition(DATABASE_DRECTORY_PATH,name_queue)
    #tcp_thread = TCPSender(SERVER_IP, SERVER_PORT, CAMERA_ID, name_queue)

    recognition_thread.start()
    #tcp_thread.start()

    #tcp_thread.join()

    logger.info("Application shutting down")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()