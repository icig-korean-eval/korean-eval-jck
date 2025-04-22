import torch
import torch.optim
import torch.utils
import torch.utils.data

import transformers

import torchaudio

from tqdm import tqdm
from evaluate import load

import pandas as pd

import os
from logger.logger import DefaultLogger
from trainer.evaluater import IPAModelEvaluator


class IPAModelTrainer:
    def __init__(self,
                 dataloader: torch.utils.data.DataLoader,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 processor: transformers.processing_utils.ProcessorMixin,
                 tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                 evaluator: IPAModelEvaluator = None,
                 device: str = 'cpu',
                 freeze_feature_extractor: bool = False,
                 total_iter: int = 200000,
                 accumulation_steps: int = 1,
                 print_steps: int = 1000,
                 eval_steps: int = 4000,
                 save_steps: int = 4000,
                 save_path: str = './save',
                 **kwargs):
        self.logger = DefaultLogger()
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.processor = processor
        self.tokenizer = tokenizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.device = device
        self.freeze_feature_extractor = freeze_feature_extractor
        self.total_iter = total_iter
        self.accumulation_steps = accumulation_steps
        self.print_steps = print_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.save_path = save_path
        
        if self.freeze_feature_extractor:
            self.model.freeze_feature_extractor()
            
        self.infinite_loader = self.__infinite_loader()
        
        self.model.to(self.device)
        self.logger.info('IPAModelTrainer init')
        
        
    def __infinite_loader(self):
        while True:
            for batch in self.dataloader:
                yield batch
                
                
    def __save_checkpoint(self, iteration=None):
        iter_name = f"iter_{iteration}" if iteration is not None else "iter_final"
        save_path = os.path.join(self.save_path, iter_name)
        os.makedirs(save_path, exist_ok=True)

        # 4. 모델과 processor 저장
        self.model.save_pretrained(os.path.join(save_path, 'model'))
        self.processor.save_pretrained(os.path.join(save_path, 'processor'))
        self.tokenizer.save_pretrained(os.path.join(save_path, 'tokenizer'))
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict()
        }, os.path.join(save_path, 'optimizer.pt'))

        self.logger.info(f"saved: {save_path}")
    
    
    def train(self):
        avg_loss = 0
        for iter in range(1, self.total_iter + 1):
            self.model.train()
            
            batch = next(self.infinite_loader)
                
            input_values = batch["input_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = self.model(input_values=input_values, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss / self.accumulation_steps
            loss.backward()
            
            if iter % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.8)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
            avg_loss += loss.item() / self.print_steps

            if iter % self.print_steps == 0:
                self.logger.debug(f"[{iter}/{self.total_iter}]\tloss: {avg_loss:.4f}\tlr: {self.lr_scheduler.get_lr()[0]}")
                avg_loss = 0
                
            if iter % self.eval_steps == 0 and self.evaluator is not None:
                self.evaluator.evaluate_cer()
                
            if iter % self.save_steps == 0:
                self.__save_checkpoint(iter)
            
            del batch["input_values"], batch["labels"], batch["attention_mask"], batch["input_lengths"], batch["label_lengths"]
        
        if self.evaluator is not None:
            self.evaluator.evaluate_cer()
        self.__save_checkpoint(iter)
