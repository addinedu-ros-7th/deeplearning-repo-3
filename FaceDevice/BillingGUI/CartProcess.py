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
    cart_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        self.cart_queue = queue.Queue()
        self.running = True

    def run(self):
        data = {"items": [{"Item": "Apple", "Count": 1, "Price": 1000},
                          {"Item": "Peach", "Count": 1, "Price": 1000}], 
                "total_price": 2000}
        print(type(data))
        self.cart_queue.put(data)
        while self.running:
            try:
                #test
                cart = self.cart_queue.get(timeout=1)
                print(cart)
                self.cart_signal.emit(cart)
                break
                
            except Exception as e:
                logger.info(f"error in CartThread.run() : {e}")
    
    def stop(self):
        self.running = False
        logger.info("Cart thread stopping")