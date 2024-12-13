import threading


from commons.logger import logger

logger = logger()

class TCPServerThread(threading.Thread):
    def __init__(self, cart_queue):
        super().__init__()
        self.cart_queue = cart_queue
        self.running = True

        logger.info("TCPServer Thread starting")

    def run(self):
        cart_data = {"items": [{"Item": "Apple", "Count": 1, "Price": 1000},
                               {"Item": "Peach", "Count": 1, "Price": 1000}], 
                    "total_price": 2000}
        while self.running:
            try:
                logger.info(f"TCP Server received cart_data : {type(cart_data)}")

                if cart_data:
                    self.cart_queue.put(cart_data)
                    logger.info(f"Put cart_data to queue: {type(cart_data)}")
                    break
                    
            except Exception as e:
                logger.info(f"Error in TCPServerThread running : {e}")
                break
    
    def stop(self):
        self.running = False
        logger.info("TCPServerThread stopping")