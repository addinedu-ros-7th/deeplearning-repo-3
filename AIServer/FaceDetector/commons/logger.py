import logging

def logger():
    logging.basicConfig(
        level=logging.DEBUG,
        #level=logging.INFO,q
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)
