from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
#from DataProcess import DataProcessorThread
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

logger = logger()

class CartThread(QThread):
    cart_signal = pyqtSignal(str)

    def __init__(self, cart_queue):
        super().__init__()
        self.cart_queue = cart_queue
        self.running = True
        logger.info(f"CartThread starting: {threading.currentThread().getName()}")

    def run(self):
        logger.info(f"CartThread running: {threading.currentThread().getName()}")
        while self.running:
            if not self.cart_queue.empty():
                cart = self.cart_queue.get(timeout=1)
                self.cart_signal.emit(cart)
                logger.info(f"Emit cart_data as cart_signal : {cart}")
                self.running = False

            else:
                #logger.info(f"cart_queue is empty.")
                time.sleep(1)
                continue
                
    def stop(self):
        self.running = False
        logger.info("Cart thread stopping")