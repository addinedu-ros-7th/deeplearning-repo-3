from deepface import DeepFace
import realtime_face_recognition
import logging
import threading
import queue
import time
import cv2
import sys
import os
import glob

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
# -----------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    #level=logging.INFO,
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
        while self.running :
            try:
                result = realtime_face_recognition.analysis(db_path=self.db_path,
                                         model_name=self.models[2],
                                         detector_backend=self.backends[3],
                                         distance_metric=self.metrics[1],
                                         time_threshold=3,
                                         name_queue=self.name_queue
                                         )
            except Exception as e:
                logger.error("Error in FaceRecognition thread: %s", e)
        
    def stop(self):
        self.running = False
        logger.info("FaceRecognition thread stopping")


class TCPSender(threading.Thread):
    def __init__(self, name_queue):
        super().__init__()
        self.name_queue = name_queue
        self.running = True
    
    def run(self):
        logger.info("TCPSender thread started: %s", threading.currentThread().getName())
        while self.running: 
            try:
                name = self.name_queue.get(timeout=1)
                logger.info("Received from queue: %s", name)
            except queue.Empty:
                logger.warning("Queue is empty.")
                continue
    
    def stop(self):
        self.running = False
        logger.info("TCPSender thread stopping")


def main():
    logger.info("Application starting")
    name_queue = queue.Queue()

    recognition_thread = FaceRecognition(DATABASE_DRECTORY_PATH,name_queue)
    tcp_thread = TCPSender(name_queue)

    recognition_thread.start()
    tcp_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupt received, stopping threads")
        recognition_thread.stop()
        tcp_thread.stop()

        recognition_thread.join()
        tcp_thread.join()

    logger.info("Application shutting down")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()