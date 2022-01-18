import logging

# consider init function to take args
default_level = logging.DEBUG
logging.basicConfig(format='%(asctime)s::%(name)s::%(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

def new_logger(name, level=default_level):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger