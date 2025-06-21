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
                 dataloader: torch.utils.data.DataLoader,  # 평가에 사용할 데이터로더
                 processor: transformers.processing_utils.ProcessorMixin,  # tokenizer + feature_extractor가 결합된 processor
                 model: torch.nn.Module,                   # 평가할 학습된 모델
                 device: str = 'cpu',                      # 평가에 사용할 디바이스
                 loop: bool = False):                      # 데이터셋이 무한 반복인지 여부
        self.logger = DefaultLogger()
        self.dataloader = dataloader
        self.processor = processor
        self.model = model
        self.device = device
        self.loop = loop
        
        # 모델을 해당 디바이스로 이동
        self.model = self.model.to(device)
    
    
    def __infinite_loader(self):
        # 무한 반복 가능한 데이터 로더 생성
        while True:
            for batch in self.dataloader:
                yield batch
    
    
    def evaluate_cer(self):
        # CER(문자 오류율)을 기준으로 모델 성능을 평가하는 함수
        cer = load("cer")  # HuggingFace evaluate 라이브러리에서 CER metric 불러오기
        self.model.eval()  # 평가 모드로 설정

        predictions = []  # 모델의 예측 텍스트
        references = []   # 정답 텍스트

        self.logger.debug(f'evaluation start\testimate count: {len(self.dataloader)}')
        iter = 1
        while True:
            # 무한 데이터 로더에서 배치 가져오기
            batch = next(self.__infinite_loader())
            
            # queue 기반 평가 데이터가 끝났음을 알리는 신호
            if not isinstance(batch, dict) and batch[0] == 'end':
                self.logger.debug('validation data finish')
                break
            input_values = batch["input_values"].to(self.device) # 입력 오디오 텐서를 디바이스로 이동
            ref_text = batch["scripts"] # 정답 스크립트 텍스트

            with torch.no_grad():
                 # 모델의 로짓 출력 얻기
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
        # 최종 평가 결과 출력
        self.logger.info(f"CER (Character Error Rate): {score:.4f}")
        
        del predictions, references
        return score
