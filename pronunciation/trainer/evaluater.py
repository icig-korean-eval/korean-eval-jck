import torch
import torchaudio
import transformers

import pandas as pd
import numpy as np
import librosa

import os
from tqdm import tqdm
from evaluate import load

from logger.logger import DefaultLogger
from utils.audio import load_soundfile


class IPAModelEvaluator:
    def __init__(self,
                 dataloader: torch.utils.data.DataLoader,
                 processor: transformers.processing_utils.ProcessorMixin,
                 model: torch.nn.Module,
                 device: str = 'cpu',
                 loop: bool = False,):
        self.logger = DefaultLogger()
        self.dataloader = dataloader
        self.processor = processor
        self.model = model
        self.device = device
        self.loop = loop
        
        self.model = self.model.to(device)
    
    
    def __infinite_loader(self):
        while True:
            for batch in self.dataloader:
                yield batch
    
    
    def evaluate_cer(self):
        cer = load("cer")
        self.model.eval()

        predictions = []
        references = []

        self.logger.debug(f'evaluation start\testimate count: {len(self.dataloader)}')
        iter = 1
        while True:
            batch = next(self.__infinite_loader())
            if not isinstance(batch, dict) and batch[0] == 'end':
                self.logger.debug('validation data finish')
                break
            input_values = batch["input_values"].to(self.device)
            ref_text = batch["scripts"]

            with torch.no_grad():
                logits = self.model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)

            pred_text = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            
            predictions.extend(pred_text)
            references.extend(ref_text)
            
            del input_values, ref_text
            if iter % 20000 == 0:
                self.logger.debug(f'[{iter}/{len(self.dataloader)}] eval')
            iter += 1
        
        score = cer.compute(predictions=predictions, references=references)
        self.logger.info(f"CER (Character Error Rate): {score:.4f}")
        
        del predictions, references
        return score
