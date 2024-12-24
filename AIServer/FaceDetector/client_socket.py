import os
import sys

import numpy as np
import socket
import json
import cv2
import time
import datetime
import base64
import threading
import queue

from commons import dir_utils
from commons.logger import logger
from modules.modeling import modeling
from recognition import RecognitionHandler

logger = logger()

class ClientThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.server_ip = "192.168.0.74"
        #self.server_ip = "192.168.45.95"
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
        self.resp = queue.Queue()
        self.recv_thread = RecvThread(self.resp, self.client_socket)
        self.recv_thread.start()
        self.running = True


    def send_images(self, frame):
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        #logger.info(f"Received frame: {type(frame)}")
        try:
            resize_frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
            result, imgencode = cv2.imencode('.jpg', resize_frame, encode_param)
            data = np.array(imgencode)
            stringData = base64.b64encode(data)
            #length = str(len(stringData))

            header = f"img|{len(stringData)}".encode('utf-8').ljust(64)

            self.client_socket.send(header)
            #self.client_socket.sendall(length.encode('utf-8').ljust(64))
            self.client_socket.send(stringData)

        except Exception as e:
            logger.error(f"Error in sending image : {e}")
            self.client_socket.close()
            time.sleep(1)
            #self.connect_server()
            #self.send_images()

            
    def send_data(self, target_id, target_name):
        #logger.info(f"Received data: {target_id} {target_name}")
        try:
            dict_data = {"member_id" : target_id, "member_name" : target_name}
            json_data = json.dumps(dict_data).encode('utf-8')

            header = f"json|{len(json_data)}".encode('utf-8').ljust(64)
    
            if self.recognition_handler.send_signal:
                self.client_socket.send(header)
                self.client_socket.send(json_data)
                logger.info(f"Data sent successfully : {json}")
        except Exception as e:
            logger.error(f"Error in sending data: {e}")
            self.client_socket.close()
            time.sleep(1)


    def run(self):
        try:
            
            while self.running:
                logger.info("Analysis is starting")
                #result = self.recognition_handler.analysis(db_path=self.db_path)
                result = self.recognition_handler.analysis(db_path=self.db_path)
                if result:
                    #logger.info(f"result : {type(result)}")
                    target_id, target_name, frame = result
                    self.send_images(frame)
                    self.send_data(target_id, target_name)
                    time.sleep(0.5)                    
                    if not self.resp.empty():
                        logger.info(">>>>>>>>>>cam stop")
                        time.sleep(5)
                        logger.info(">>>>>>>>>>>>>cam start")
                        resp = self.resp.get(timeout=1)
                        logger.info(">>>>>>>>>>> get queue")

                else :
                    logger.warning("No image returned from analysis")
                    continue

        except Exception as e:
            logger.error("Error in ClientThread running: %s", e)
            self.client_socket.close()
            logger.info("ClientThread has stopped")
    
    def stop(self):
        if self.recv_thread.running:
            self.recv_thread.stop()
        self.recognition_handler.close()
        self.running = False
        self.client_socket.close()
        logger.info(f"ClientThread has stopped")


class RecvThread(threading.Thread):
    def __init__(self, resp, client_socket):
        super().__init__()
        self.client_socket = client_socket
        self.resp = resp
        self.running = True

    def run(self):
        logger.info("RecvThread is starting")
        while self.running:
            try:
                result = self.client_socket.recv(1024).decode()
                if result is None:
                    continue
                self.resp.put(result)
            except Exception as e:
                #logger.info(f"Error receiving data : {e}")
                continue
        #self.exec_()

    def stop(self):
        self.running = False
        logger.info("RecvThread has been stopped")


def main():
    client_thread = ClientThread()
    client_thread.start()

if __name__ == "__main__":
    main()