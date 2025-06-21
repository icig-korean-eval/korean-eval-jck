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
                 dataloader: torch.utils.data.DataLoader,  # 학습 데이터 로더
                 model: torch.nn.Module,                   # 학습할 모델 (Wav2Vec2)
                 optimizer: torch.optim.Optimizer,         # 최적화 알고리즘 (예: AdamW)
                 processor: transformers.processing_utils.ProcessorMixin,  # 전처리 도구 (tokenizer + feature_extractor)
                 tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,  # tokenizer 객체
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,  # learning rate scheduler
                 evaluator: IPAModelEvaluator = None,       # 평가 클래스 (optional)
                 device: str = 'cpu',                       # 학습 디바이스 설정
                 freeze_feature_extractor: bool = False,    # feature extractor freezing 여부
                 total_iter: int = 200000,                  # 총 학습 iteration 수
                 accumulation_steps: int = 1,               # gradient accumulation step 수
                 print_steps: int = 1000,                   # 로그 출력 주기
                 eval_steps: int = 4000,                    # 평가 주기
                 save_steps: int = 4000,                    # 체크포인트 저장 주기
                 save_path: str = './save',                 # 모델 저장 디렉토리
                 loop: bool = False,                        # 데이터 반복 여부 (infinite mode)
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
        self.loop = loop
        
        # feature extractor를 freeze 시킬 경우
        if self.freeze_feature_extractor:
            self.model.freeze_feature_extractor()
            
        # 반복 가능한 무한 데이터 로더 설정
        self.infinite_loader = self.__infinite_loader()
        
        self.model.to(self.device)
        self.logger.info('IPAModelTrainer init')
        
        
    def __infinite_loader(self):
        # 무한히 데이터를 제공하는 반복자 생성
        while True:
            for batch in self.dataloader:
                yield batch
                
                
    def __save_checkpoint(self, iteration=None):
        # 체크포인트 저장 함수
        iter_name = f"iter_{iteration}" if iteration is not None else "iter_final"
        save_path = os.path.join(self.save_path, iter_name)
        os.makedirs(save_path, exist_ok=True)

        # 모델, processor, tokenizer, optimizer 저장
        self.model.save_pretrained(os.path.join(save_path, 'model'))
        self.processor.save_pretrained(os.path.join(save_path, 'processor'))
        self.tokenizer.save_pretrained(os.path.join(save_path, 'tokenizer'))
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict()
        }, os.path.join(save_path, 'optimizer.pt'))

        self.logger.info(f"saved: {save_path}")
    
    
    def __get_batch_data(self, batch):
        # 배치에서 텐서들을 디바이스로 이동시킴
        input_values = batch["input_values"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        return input_values, labels, attention_mask
    
    
    def __update_model(self, input_values, attention_mask, labels, iter, last_batch=False, last_batch_section=False, remain=1):
        # 모델 학습 및 gradient accumulation 수행
        outputs = self.model(input_values=input_values, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # 마지막 누락된 배치일 경우 보정
        if not self.loop and last_batch_section:
            loss = loss / remain
        else:
            loss = loss / self.accumulation_steps
        loss.backward()
        
        # gradient clipping 및 optimizer step
        if iter % self.accumulation_steps == 0 or (last_batch and last_batch_section):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.8)
            self.optimizer.step()
            # if self.loop:
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        return loss.item() / self.print_steps
    
    
    def __save_best_cer(self, cer, iter):
        # CER(문자 오류율)이 더 낮으면 베스트 모델로 저장
        if self.best_cer > cer:
            self.__save_checkpoint(iter)
            self.best_cer = cer
    
    
    def train(self):
        # 학습 전체 루프 정의
        avg_loss = 0
        self.best_cer = 0.5
        if self.loop:
            # 무한 반복 학습 방식
            for iter in range(1, self.total_iter + 1):
                self.model.train()
                
                batch = next(self.infinite_loader)
                    
                input_values, labels, attention_mask = self.__get_batch_data(batch)

                loss = self.__update_model(input_values, attention_mask, labels, iter)
                avg_loss += loss

                if iter % self.print_steps == 0:
                    self.logger.debug(f"[{iter}/{self.total_iter}]\tloss: {avg_loss:.4f}\tlr: {self.lr_scheduler.get_lr()[0]}")
                    avg_loss = 0
                    
                if iter % self.eval_steps == 0 and self.evaluator is not None:
                    self.evaluator.evaluate_cer()
                    
                if iter % self.save_steps == 0:
                    self.__save_checkpoint(iter)
                
                del batch["input_values"], batch["labels"], batch["attention_mask"], batch["input_lengths"], batch["label_lengths"]
        else:
            # epoch 기반 반복 학습 방식
            for epoch in range(1, self.total_iter + 1):
                for iter, batch in tqdm(enumerate(self.dataloader, start=1), total=len(self.dataloader), ncols=0, desc='train'):
                    last_batch = iter == len(self.dataloader)
                    last_batch_section = iter > (len(self.dataloader) // self.accumulation_steps * self.accumulation_steps)
                    self.model.train()
                        
                    input_values, labels, attention_mask = self.__get_batch_data(batch)

                    loss = self.__update_model(input_values, attention_mask, labels, iter, last_batch, last_batch_section, len(self.dataloader) % self.accumulation_steps)
                    avg_loss += loss

                    if iter % self.print_steps == 0:
                        self.logger.debug(f"[{epoch}/{self.total_iter}] [{iter}/{len(self.dataloader)}]\tloss: {avg_loss:.4f}\tlr: {self.lr_scheduler.get_lr()[0]}")
                        avg_loss = 0
                    
                    del batch["input_values"], batch["labels"], batch["attention_mask"], batch["input_lengths"], batch["label_lengths"]
                    
                # if not self.loop:
                #     self.lr_scheduler.step()
                
                # epoch 끝마다 CER 평가 후 best model 저장
                cer = self.evaluator.evaluate_cer()
                self.__save_best_cer(cer, epoch)
        
        # 마지막 평가 및 최종 체크포인트 저장
        if self.evaluator is not None:
            self.evaluator.evaluate_cer()
        self.__save_checkpoint()
