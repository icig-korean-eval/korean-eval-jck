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
                 data_path: str,
                 sound_path_prefix: str,
                 processor: Wav2Vec2Processor,
                 batch_size: int,
                 x: typing.Optional[str],
                 y: typing.Optional[str],
                 max_queue_size: int=64,
                 refill_threshold: int=16,
                 chunk_size: int=64,
                 notify_end=False):
        super().__init__()
        self.logger = DefaultLogger()
        self.processor = processor
        self.max_queue_size = max_queue_size
        self.refill_threshold = refill_threshold
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        
        self.label_x = x
        self.label_y = y
        self.sound_path_prefix = sound_path_prefix

        self.data_index = self._load_csv(data_path)
        self.data_len = len(self.data_index)
        self.q = queue.Queue(maxsize=self.max_queue_size)
        self.cursor = 0
        self.lock = threading.Lock()
        
        self.notify_end = notify_end

        self._start_loading_thread()

    def _load_csv(self, path: str) -> typing.List:
        df = pd.read_pickle(path)
        
        if self.label_y is not None:
            df["label_len"] = df[self.label_y].apply(len)
            df = df[df['label_len'] > 10].sort_values("label_len").drop(columns=["label_len"])
        
        total = len(df)
        remainder = total % self.batch_size
        if remainder != 0:
            df = df.iloc[:total - remainder].reset_index(drop=True)
        
        self.logger.info(f'data size: {df.shape}')
        
        df = df[[self.label_x] + [self.label_y]]
        
        return list(df.itertuples(index=False, name=None))

    def _load_data(self, file: str,
                   text: typing.List[str] | None) -> typing.Optional[typing.Dict]:
        data = dict()
        
        waveform = load_soundfile(file)
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
            if input_values.shape[0] < labels.shape[0]:
                return
        return data

    def _refill_queue(self):
        self.logger.debug(f'refill data: {self.cursor}\tapproximate remain: {self.q.qsize()}')
        with self.lock:
            end = min(self.cursor + self.chunk_size, self.data_len)
            for i in range(self.cursor, end):
                item = self._load_data(
                    file=os.path.join(self.sound_path_prefix, self.data_index[i][0]),
                    text=self.data_index[i][1]
                )
                if item is None:
                    self.logger.error(f'length error - idx: {i}')
                    continue
                self.q.put(item)
            if end < self.data_len:
                self.cursor = end
            else:
                self.cursor = 0
                if self.notify_end:
                    self.q.put('end')

    def _start_loading_thread(self):
        def _run():
            while True:
                if self.q.qsize() <= self.refill_threshold:
                    self._refill_queue()
                time.sleep(0.5)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def __iter__(self):
        while True:
            item = self.q.get()
            yield item
            del item
            
    def __len__(self):
        return self.data_len
    