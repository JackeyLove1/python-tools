import os
import logging

# create the logger object
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create a formatter for the log messages
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

# create a StreamHandler to write the log messages to the console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# create a FileHandler to write the log messages to a file in the data folder
log_folder = 'log'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
log_file = os.path.join(log_folder, 'project.log')
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)

# add the handlers to the logger object
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# use loguru
# pip install loguru
# https://github.com/Delgan/loguru
from loguru import logger
import sys
logger.debug("That's it, beautiful and simple logging!")
# set cout or fileand format
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
# add log file
logger.add("file_{time}.log")
logger.add("file_1.log", rotation="500 MB")    # Automatically rotate too big file
logger.add("file_2.log", rotation="12:00")     # New file is created each day at noon
logger.add("file_3.log", rotation="1 week")    # Once the file is too old, it's rotated
logger.add("file_X.log", retention="10 days")  # Cleanup after some time
logger.add("file_Y.log", compression="zip")    # Save some loved space
# set color
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")
# async
logger.add("somefile.log", enqueue=True)

