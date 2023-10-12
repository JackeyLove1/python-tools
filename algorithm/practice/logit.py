import logging
import os
import datetime
from logging.handlers import RotatingFileHandler
log_directory = os.getcwd()
log_file = "mylog"
max_bytes = 100 * 1024 * 1024 # 100M
backups = 10
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
log_file = os.path.join(log_directory, '{}_{}.log'.format(log_file, timestamp))
file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backups)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s (%(funcName)s:%(lineno)d)] %(levelname)s in %(module)s: %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Test the logger
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")