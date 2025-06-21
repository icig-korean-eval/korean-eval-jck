import os
import sys
import logging


class DefaultLogger:
    # 클래스 변수로 싱글톤 인스턴스를 저장
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        # 인스턴스가 없을 경우에만 새로 생성 (싱글톤 패턴)
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,
                 name: str = 'main',
                 stream: bool = True,
                 file: bool = False,
                 path: str = './log'):
        # 이미 초기화가 끝났다면 다시 초기화하지 않음
        if self._initialized: return

        self.logger_name = name

        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.DEBUG)

        # 콘솔 출력 핸들러 설정
        if stream:
            formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # 파일 출력 핸들러 설정
        if file:
            formatter_file = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
            handler_file = logging.FileHandler(os.path.join(path, f'{self.logger_name}.log'))
            handler_file.setLevel(logging.DEBUG)
            handler_file.setFormatter(formatter_file)
            self.logger.addHandler(handler_file)

        self._initialized = True
        
        # 예외 발생 시 로그로 기록하는 예외 훅 설정
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
        # 멀티 GPU 분산 학습 시 로그 출력 여부 결정 함수 (현재 무조건 True 반환)
        # 현재는 멀티GPU를 사용하지 않으므로 항상 True
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
