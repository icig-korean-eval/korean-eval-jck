from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Config
from transformers import TrainingArguments, Trainer, get_linear_schedule_with_warmup

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import PreTokenizer
from transformers import PreTrainedTokenizerFast

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset, DataLoader

import torchaudio
import pandas as pd

import threading
import queue

import json
import os
import time
import datetime

from tqdm import tqdm
from evaluate import load


from typing import List, Optional, Union
import numpy as np
from transformers.tokenization_utils import PreTrainedTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import AddedToken, BatchEncoding
from transformers.utils import (
    ModelOutput,
    PaddingStrategy,
    TensorType,
    add_end_docstrings,
    is_flax_available,
    is_tf_available,
    is_torch_available,
    logging,
    to_py_obj,
)

import logging


start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logger = logging.getLogger('ctc')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

os.makedirs(os.path.join('save', start_time), exist_ok=True)
path = os.path.join('save', start_time, f'logging.log')
file_handler = logging.FileHandler(path, encoding='utf-8')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

        
class StreamingDatasetQueue(IterableDataset):
    def __init__(self, data_path, processor, batch_size, max_queue_size=64, refill_threshold=16, chunk_size=64):
        super().__init__()
        self.processor = processor
        self.max_queue_size = max_queue_size
        self.refill_threshold = refill_threshold
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        self.data_index = self._load_csv(data_path)
        self.data_len = len(self.data_index)
        self.q = queue.Queue(maxsize=self.max_queue_size)
        self.cursor = 0  # 현재 index
        self.lock = threading.Lock()

        # 백그라운드에서 데이터를 계속 넣는 쓰레드 실행
        self._start_loading_thread()

    def _load_csv(self, path):
        df = pd.read_pickle(path)  # file, text
        df["label_len"] = df["script.text.ipa"].apply(len)
        df = df.sort_values("label_len").drop(columns=["label_len"])
        
        total = len(df)
        remainder = total % self.batch_size
        if remainder != 0:
            df = df.iloc[:total - remainder].reset_index(drop=True)
        
        logger.info(f'data size: {df.shape}')
        return list(df.itertuples(index=False, name=None))  # [(file, text), ...]

    def _load_data(self, file, text):
        waveform, sr = torchaudio.load(file)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        input_values = self.processor.feature_extractor(
            waveform.squeeze().numpy(), sampling_rate=16000
        ).input_values[0]
        labels = self.processor.tokenizer(text).input_ids
        
        input_values = torch.tensor(input_values, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long).squeeze()
        # labels = self.encoder.encode(text)
        
        # logger.info(f'{self.processor.tokenizer.decode(labels)} {text}')
        
        if input_values.shape[0] < labels.shape[0]:
            return
        
        return {
            "input_values": input_values,
            "labels": labels
        }

    def _refill_queue(self):
        logger.debug(f'refill data: {self.cursor}\tapproximate remain: {self.q.qsize()}')
        with self.lock:
            end = min(self.cursor + self.chunk_size, self.data_len)
            for i in range(self.cursor, end):
                item = self._load_data(
                    file=os.path.join('../data/announcer/source', self.data_index[i][0]),
                    text=self.data_index[i][2]
                )
                if item is None:
                    logger.error(f'length error - idx: {i}')
                    continue
                self.q.put(item)
            self.cursor = end if end < self.data_len else 0  # 순환 가능

    def _start_loading_thread(self):
        def _run():
            while True:
                if self.q.qsize() <= self.refill_threshold:
                    self._refill_queue()
                time.sleep(0.5)  # 너무 자주 체크하지 않도록

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def __iter__(self):
        while True:
            item = self.q.get()
            yield item
            del item  # 명시적으로 제거
            
            
class IPADataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        input_values = [item["input_values"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_lengths = [len(x) for x in input_values]
        label_lengths = [len(x) for x in labels]
        
        input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=self.processor.feature_extractor.padding_value)
        # labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        attention_mask = torch.zeros_like(input_values_padded).long()
        for i, l in enumerate(input_lengths):
            attention_mask[i, :l] = 1

        return {
            "input_values": input_values_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "input_lengths": torch.tensor(input_lengths),
            "label_lengths": torch.tensor(label_lengths)
        }


def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch
            
            
def evaluate_cer(model, processor, eval_csv_path):
    cer = load("cer")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_pickle(eval_csv_path)

    predictions = []
    references = []

    for file, ref_text, _, _ in tqdm(df.itertuples(index=False), desc='eval', total=df.shape[0]):
        waveform, sr = torchaudio.load(os.path.join('../data/announcer/source', file))
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        input_values = processor.feature_extractor(
            waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]

        pred_text = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
        predictions.append(pred_text)
        references.append(ref_text)
        
        del file, ref_text
        
    score = cer.compute(predictions=predictions, references=references)
    logger.debug(f"CER (Character Error Rate): {score:.4f}")
    return score


def save_checkpoint(model, start_time, processor, optimizer, tokenizer, lr_scheduler, save_root="save", iteration=None):
    # 1. 날짜 기반 폴더명 생성
    date_folder = os.path.join(save_root, start_time)

    # 2. 날짜 폴더 없으면 생성
    if not os.path.exists(date_folder):
        os.makedirs(date_folder)

    # 3. iter 폴더 이름 설정
    iter_name = f"iter_{iteration}" if iteration is not None else "iter_final"
    save_path = os.path.join(date_folder, iter_name)
    os.makedirs(save_path, exist_ok=True)

    # 4. 모델과 processor 저장
    model.save_pretrained(os.path.join(save_path, 'model'))
    processor.save_pretrained(os.path.join(save_path, 'processor'))
    tokenizer.save_pretrained(os.path.join(save_path, 'tokenizer'))
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }, os.path.join(save_path, 'optimizer.pt'))

    logger.debug(f"saved: {save_path}")
    return save_path


if __name__ == "__main__":
    batch_size = 10
    total_iter = 20 * 30000 // batch_size
    accumulation_steps = 64 // batch_size
    print_step = 1600 // batch_size
    eval_step = 24000 // batch_size
    save_step = 24000 // batch_size
    
    datas = pd.read_pickle('../data/announcer/labeling/preprocessed.pickle')
    logger.info(datas.head())

    tokenizer = Wav2Vec2CTCTokenizer("./model/ipa_vocab_auto.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" ")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=False, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    logger.info(f'tokens: {tokenizer.get_vocab()}')
    
    dataset = StreamingDatasetQueue(
        data_path='../data/announcer/labeling/preprocessed_train.pickle',
        processor=processor,
        max_queue_size=50000,
        refill_threshold=1000,
        chunk_size=40000,
        batch_size=batch_size
    )
    
    collator = IPADataCollator(processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collator,
        drop_last=True
    )
    
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        vocab_size=len(tokenizer),
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.005)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=30000 * 2 // accumulation_steps // batch_size, num_training_steps=total_iter//accumulation_steps)
    
    avg_loss = 0
    
    train_loader = infinite_loader(dataloader)
    
    model.freeze_feature_extractor()
    
    for iter in range(1, total_iter + 1):
        model.train()
        
        batch = next(train_loader)
            
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # optimizer.zero_grad()
        outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss = loss / accumulation_steps
        loss.backward()
        
        if iter % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        avg_loss += loss.item() / print_step

        if iter % print_step == 0:
            logger.debug(f"[{iter}/{total_iter}]\tloss: {avg_loss:.4f}\tlr: {lr_scheduler.get_lr()[0]}")
            avg_loss = 0
            
        if iter % eval_step == 0:
            evaluate_cer(model, processor, '../data/announcer/labeling/preprocessed_test.pickle')
            
        if iter % save_step == 0:
            save_checkpoint(model, start_time, processor, optimizer, tokenizer, lr_scheduler, iteration=iter)
        
        del batch["input_values"], batch["labels"], batch["attention_mask"], batch["input_lengths"], batch["label_lengths"]
