from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

from commons.logger import logger

logger.logger()

class CartThread(QThread):
    cart_signal = pyqtSignal(dict)

    def __init__(self, cart_queue):
        self.cart_queue = cart_queue
        self.running = True

    def run(self):
        while self.running:
            try:
                cart_data = self.cart_queue.get(timeout=1)
                if cart_data:
                    self.cart_queue.put(cart_data)
                    logger.info(f"Put cart_data to queue : {cart_data}")
                    self.cart_signal.emit(cart_data)
                    logger.info(f"Emit cart_data as cart_signal : {cart_data}")
                    break
                
            except Exception as e:
                logger.info(f"error in CartThread running : {e}")
                self.running = False
    
    def stop(self):
        self.running = False
        logger.info("CartThread stopping")