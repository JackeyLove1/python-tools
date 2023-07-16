import logging
import threading
from logging.handlers import RotatingFileHandler


class SingletonLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter("[%(asctime)s (%(funcName)s:%(lineno)d)] %(levelname)s in %(module)s: %(message)s")
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger


logger = SingletonLogger().get_logger()
