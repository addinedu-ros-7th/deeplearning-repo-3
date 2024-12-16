import os
import sys

import numpy as np
import socket
import cv2
import time
import datetime
import base64
import threading

from commons import dir_utils
from commons.logger import logger
from modules.modeling import modeling
from recognition import RecognitionHandler

logger = logger()

class ClientSocket(threading.Thread):
    def __init__(self):
        super().__init__()
        #self.server_ip = "192.168.0.74"
        self.server_ip = "192.168.45.95"
        self.server_port = 5005

        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.server_ip, self.server_port))
            logger.info("Client has been connected to server")
        except Exception as e:
            print(e)
            self.client_socket.close()
        
        self.db_path = dir_utils.initialize_dir()
        self.recognition_handler = RecognitionHandler(
            db_path=self.db_path,
            model_name=modeling["models"][2],
            detector_backend=modeling["backends"][4],
            distance_metric=modeling["metrics"][1],
            time_threshold=3
            )
        self.running = True


    def send_images(self, frame):
        #capture = cv2.VideoCapture(0)
        #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        #logger.info(f"Received frmae: {type(frame)}")
        try:
            resize_frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
            result, imgencode = cv2.imencode('.jpg', resize_frame, encode_param)
            data = np.array(imgencode)
            stringData = base64.b64encode(data)
            length = str(len(stringData))
            #logger.info("Try to send data to server")
            self.client_socket.sendall(length.encode('utf-8').ljust(64))
            self.client_socket.send(stringData)
            #logger.info("string data has been send")
            #logger.info("current frame has been send")"""
            """
            while capture.isOpened():
                ret, frame = capture.read()
                resize_frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
                
                encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
                result, imgencode = cv2.imencode('.jpg', resize_frame, encode_param)
                data = np.array(imgencode)
                stringData = base64.b64encode(data)
                length = str(len(stringData))

                #logger.info("Try to send data to server")
                self.client_socket.sendall(length.encode('utf-8').ljust(64))
                self.client_socket.send(stringData)
                #logger.info("string data has been send")
                #logger.info("current frame has been send")"""

        except Exception as e:
            logger.error(f"Error in sending image : {e}")
            self.client_socket.close()
            time.sleep(1)
            #self.connect_server()
            #self.send_images()
    
    def run(self):
        try:
            logger.info("Performing analysis")
            while self.running:
                
                result = self.recognition_handler.analysis(db_path=self.db_path)
                if result is not None:
                    #logger.info(f"result : {type(result)}")
                    self.send_images(result)
                else :
                    logger.warning("No image returned from analysis")
                    continue

        except Exception as e:
            logger.error("Error in CameraThread: %s", e)
            self.client_socket.close()
            logger.info("ClientThread has stopped")
    
    def stop(self):
        self.running = False
        logger.info(f"ClientThread has stopped")


def main():
    client_thread = ClientSocket()
    client_thread.start()

if __name__ == "__main__":
    main()