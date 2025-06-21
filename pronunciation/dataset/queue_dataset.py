import torchaudio
import torch
from torch.utils.data import IterableDataset

from transformers import Wav2Vec2Processor

import pandas as pd

import os
import queue
import threading
import time

from logger.logger import DefaultLogger
from utils.audio import load_soundfile

import typing


class StreamingDatasetQueue(IterableDataset):
    def __init__(self,
                 data_path: str,                 # pickle로 저장된 메타데이터(csv) 경로
                 sound_path_prefix: str,         # 오디오 파일 경로 앞부분
                 processor: Wav2Vec2Processor,   # Wav2Vec2 processor (토크나이저 + feature extractor)
                 batch_size: int,                # 배치 크기
                 x: typing.Optional[str],        # 오디오 파일명 열 이름
                 y: typing.Optional[str],        # 정답 스크립트 열 이름
                 max_queue_size: int=64,         # 큐 최대 크기
                 refill_threshold: int=16,       # refill이 일어나는 큐의 하한 임계값
                 chunk_size: int=64,             # refill 시 로드할 데이터 개수
                 notify_end=False,               # 루프 끝에 'end' 토큰 삽입 여부
                 loop=False):                    # 전체 데이터를 계속 반복할지 여부
        super().__init__()
        self.logger = DefaultLogger()
        self.loop = loop
        
        self.processor = processor
        self.max_queue_size = max_queue_size
        self.refill_threshold = refill_threshold
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        
        self.label_x = x
        self.label_y = y
        self.sound_path_prefix = sound_path_prefix

        # 메타데이터 로드 (pickle 파일)
        self.data_index = self._load_csv(data_path)
        self.data_len = len(self.data_index)
        
        # 데이터 큐 및 제어 변수 초기화
        self.q = queue.Queue(maxsize=self.max_queue_size)
        self.cursor = 0
        self.lock = threading.Lock()
        
        self.notify_end = notify_end

        # 데이터를 백그라운드에서 로드할 스레드 시작
        self._start_loading_thread()

    def _load_csv(self, path: str) -> typing.List:
        # pickle 파일로부터 DataFrame 로드
        df = pd.read_pickle(path)
        
        # 정답 텍스트(y)가 있으면, 길이가 10 이상인 샘플만 필터링
        if self.label_y is not None:
            df["label_len"] = df[self.label_y].apply(len)
            df = df[df['label_len'] > 10].sort_values("label_len").drop(columns=["label_len"])
        
        # 루프 학습 시 배치 크기에 맞게 나머지 데이터 제거 (batch_size의 배수로 맞춤
        total = len(df)
        if self.loop:
            remainder = total % self.batch_size
            if remainder != 0:
                df = df.iloc[:total - remainder].reset_index(drop=True)
        # df = df.iloc[:4043].reset_index(drop=True)
        
        self.logger.info(f'data size: {df.shape}')
        
        df = df[[self.label_x] + [self.label_y]]
        
        return list(df.itertuples(index=False, name=None))

    def _load_data(self, file: str,
                   text: typing.List[str] | None) -> typing.Optional[typing.Dict]:
        data = dict()
        
        # 오디오 파일 로드
        waveform = load_soundfile(file)
        # feature extractor 적용
        input_values = self.processor.feature_extractor(
            waveform.squeeze().numpy(), sampling_rate=16000
        ).input_values[0]
        
        input_values = torch.tensor(input_values, dtype=torch.float)
        data['input_values'] = input_values
        
        if text is not None:
            data['script'] = ''.join(text)
            labels = self.processor.tokenizer(text).input_ids
            labels = torch.tensor(labels, dtype=torch.long).squeeze()
            
            data['labels'] = labels
            # input보다 label이 더 긴 경우는 오류로 간주
            if input_values.shape[0] < labels.shape[0]:
                return
        return data

    def _refill_queue(self):
        # 큐가 임계 이하일 때 새 데이터를 큐에 채우는 함수
        with self.lock:
            end = min(self.cursor + self.chunk_size, self.data_len)
            if self.cursor == end:
                return
            self.logger.debug(f'refill data: {self.cursor}\tapproximate remain: {self.q.qsize()}')
            
            # 지정된 범위만큼 데이터를 로드하고 큐에 추가
            for i in range(self.cursor, end):
                item = self._load_data(
                    file=os.path.join(self.sound_path_prefix, self.data_index[i][0]),
                    text=self.data_index[i][1]
                )
                if item is None:
                    self.logger.error(f'length error - idx: {i}')
                    continue
                self.q.put(item)
                
            # 루프 모드일 경우 끝에 도달하면 다시 처음부터 시작
            if self.loop:
                if end < self.data_len:
                    self.cursor = end
                else:
                    self.cursor = 0
                    if self.notify_end:
                         # 루프 종료 알림 토큰 삽입
                        self.q.put('end')
            else:
                self.cursor = end

    def _start_loading_thread(self):
        def _run():
            while True:
                if self.q.qsize() <= self.refill_threshold:
                    self._refill_queue()
                time.sleep(0.5)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        
    def _reset(self):
        # 평가 모드 등에서 끝나고 다시 초기화할 때 호출
        with self.lock:
            self.cursor = 0
            self.q = queue.Queue(maxsize=self.max_queue_size)

    def __iter__(self):
        # 파이토치의 IterableDataset 반복자 정의
        if self.loop:
            while True:
                item = self.q.get()
                yield item
                del item
        else:
            try:
                for _ in range(self.data_len):
                    item = self.q.get()
                    yield item
                    del item
            finally:
                self._reset()
            
    def __len__(self):
        return self.data_len
    