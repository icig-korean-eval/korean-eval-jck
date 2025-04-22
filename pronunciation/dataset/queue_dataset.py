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

import typing


class StreamingDatasetQueue(IterableDataset):
    def __init__(self,
                 data_path: str,
                 processor: Wav2Vec2Processor,
                 batch_size: int,
                 max_queue_size: int=64,
                 refill_threshold: int=16,
                 chunk_size: int=64):
        super().__init__()
        self.logger = DefaultLogger()
        self.processor = processor
        self.max_queue_size = max_queue_size
        self.refill_threshold = refill_threshold
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        self.data_index = self._load_csv(data_path)
        self.data_len = len(self.data_index)
        self.q = queue.Queue(maxsize=self.max_queue_size)
        self.cursor = 0
        self.lock = threading.Lock()

        self._start_loading_thread()

    def _load_csv(self, path: str) -> typing.List:
        df = pd.read_pickle(path)  # file, text
        df["label_len"] = df["script.text.ipa"].apply(len)
        df = df.sort_values("label_len").drop(columns=["label_len"])
        
        total = len(df)
        remainder = total % self.batch_size
        if remainder != 0:
            df = df.iloc[:total - remainder].reset_index(drop=True)
        
        self.logger.info(f'data size: {df.shape}')
        return list(df.itertuples(index=False, name=None))  # [(file, text), ...]

    def _load_data(self, file: str,
                   text: typing.List[str]) -> typing.Optional[typing.Dict]:
        waveform, sr = torchaudio.load(file)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        input_values = self.processor.feature_extractor(
            waveform.squeeze().numpy(), sampling_rate=16000
        ).input_values[0]
        labels = self.processor.tokenizer(text).input_ids
        
        input_values = torch.tensor(input_values, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long).squeeze()
        
        if input_values.shape[0] < labels.shape[0]:
            return
        
        return {
            "input_values": input_values,
            "labels": labels
        }

    def _refill_queue(self):
        self.logger.debug(f'refill data: {self.cursor}\tapproximate remain: {self.q.qsize()}')
        with self.lock:
            end = min(self.cursor + self.chunk_size, self.data_len)
            for i in range(self.cursor, end):
                item = self._load_data(
                    file=os.path.join('../data/announcer/source', self.data_index[i][0]),
                    text=self.data_index[i][2]
                )
                if item is None:
                    self.logger.error(f'length error - idx: {i}')
                    continue
                self.q.put(item)
            self.cursor = end if end < self.data_len else 0

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
    