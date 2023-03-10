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