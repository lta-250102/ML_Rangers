import logging


def init_logger():
    '''init logger'''
    logger = logging.getLogger("ml_ranger_logger")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s \t %(levelname)s \t %(message)s')

    logger.handlers = []

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler('error.log')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
