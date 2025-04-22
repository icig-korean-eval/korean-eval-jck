import os
import sys
import logging


class DefaultLogger:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,
                 name: str,
                 stream: bool,
                 file: bool,
                 path: str):
        if self._initialized: return

        self.logger_name = name

        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.DEBUG)

        if stream:
            formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        if file:
            formatter_file = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
            handler_file = logging.FileHandler(os.path.join(path, f'{self.logger_name}.log'))
            handler_file.setLevel(logging.DEBUG)
            handler_file.setFormatter(formatter_file)
            self.logger.addHandler(handler_file)

        self._initialized = True
        
        def catch_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            logger = logging.getLogger(self.logger_name)

            logger.error(
                "Unexpected exception.",
                exc_info=(exc_type, exc_value, exc_traceback)
            )
        sys.excepthook = catch_exception
        
        
    def __check_gpu_rank(self, gpu_rank: int) -> bool:
        return True
        if self.parallel == 0:
            return True

        if gpu_rank == 0:
            return True
        return False


    def debug(self,
              msg: str,
              gpu_rank: int = -1):
        if not self.__check_gpu_rank(gpu_rank):
            return
        self.logger.debug(msg)


    def info(self,
              msg: str,
              gpu_rank: int = -1):
        if not self.__check_gpu_rank(gpu_rank):
            return
        self.logger.info(msg)


    def warning(self,
              msg: str,
              gpu_rank: int = -1):
        if not self.__check_gpu_rank(gpu_rank):
            return
        self.logger.warning(msg)


    def error(self,
              msg: str,
              gpu_rank: int = -1):
        if not self.__check_gpu_rank(gpu_rank):
            return
        self.logger.error(msg)


    def exception(self,
              msg: str,
              gpu_rank: int = -1):
        if not self.__check_gpu_rank(gpu_rank):
            return
        self.logger.exception(msg)
