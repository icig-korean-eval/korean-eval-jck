from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    get_linear_schedule_with_warmup
)
import torch
from torch.utils.data import DataLoader
from dataset.queue_dataset import StreamingDatasetQueue
from dataset.collator import IPADataCollator

from trainer.trainer import IPAModelTrainer
from trainer.evaluater import IPAModelEvaluator

import os
import yaml
import argparse
import datetime
import math

from logger.logger import DefaultLogger


def parse_config():
    # 인자를 파싱하기 위한 ArgumentParser 생성
    parser = argparse.ArgumentParser()
    # 실행 모드: 학습(train) 또는 테스트(test)를 선택 (기본값은 'train')
    parser.add_argument('--mode', type=str, default='train', help='train: 학습, test: 테스트')
    # GPU 사용 여부를 지정하는 플래그 (입력 시 True)
    parser.add_argument('--gpu', action='store_true', help='gpu')
    # 사전 학습된 모델 경로를 전달받을 수 있는 인자
    parser.add_argument('--pretrained_path', type=str, help='pretrained teset')
    args = parser.parse_args()
    
    # 모드에 따라 사용할 yaml 설정 파일 경로 지정
    if args.mode == 'train':
        yaml_path = './config/train.yaml'
    elif args.mode == 'test':
        yaml_path = './config/test.yaml'
    else:
        raise argparse.ArgumentError(
            message=f'mode는 train, test만 가능합니다.'
        )
    
     # yaml 설정 파일 열기
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 현재 실행 시간을 저장
    config['excute_time'] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 학습 결과 저장 디렉토리 경로 설정
    config['save_path'] = os.path.join('./save', config['excute_time'])
    if args.mode == 'test':
        config['save_path'] = os.path.join('./save/test', config['excute_time'])
    os.makedirs(config['save_path'], exist_ok=True)
    
    # 로그 저장 경로 설정
    config['log']['path'] = config['save_path']
    
    with open(os.path.join(config['save_path'], 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f)
    
    # 모드/device 정보를 config에 저장
    config['mode'] = args.mode
    config['device'] = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    
    return config


if __name__ == "__main__":
    config = parse_config()
    # 로거 설정
    logger = DefaultLogger(**config['log'])
    logger.info(f'condig: {config}')
    
    if config['mode'] == 'train':
        # 무한 반복 학습일 경우, 반복 횟수 등을 배치 사이즈로 나눠서 step 기준으로 변환
        if config['train_dataset']['loop']:
            config['total_iter'] //= config['batch_size']
            config['eval_steps'] //= config['batch_size']
            config['save_steps'] //= config['batch_size']
        else:
            pass
        config['print_steps'] //= config['batch_size']
        config['accumulation_steps'] //= config['batch_size']
        
        # 토크나이저 및 피처 추출기 정의
        tokenizer = Wav2Vec2CTCTokenizer(config['vocab_path'], unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" ")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=False, return_attention_mask=False)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        
        logger.info(f'tokens: {tokenizer.get_vocab()}')
        
        # 학습 데이터셋 정의 (Streaming 방식으로 메모리 절약)
        dataset_train = StreamingDatasetQueue(
            data_path=config['train_dataset']['path'],
            sound_path_prefix=config['train_dataset']['audio_path'],
            processor=processor,
            x=config['train_dataset']['x'],
            y=config['train_dataset']['y'],
            max_queue_size=config['max_queue_size'],
            refill_threshold=config['refill_threshold'],
            chunk_size=config['chunk_size'],
            batch_size=config['batch_size'],
            loop=config['train_dataset']['loop'],
        )
        # 평가용 데이터셋 정의
        dataset_eval = StreamingDatasetQueue(
            data_path=config['eval_dataset']['path'],
            sound_path_prefix=config['eval_dataset']['audio_path'],
            processor=processor,
            x=config['eval_dataset']['x'],
            y=config['eval_dataset']['y'],
            max_queue_size=config['max_queue_size'],
            refill_threshold=config['refill_threshold'],
            chunk_size=config['chunk_size'],
            batch_size=1,
            notify_end=True,
            loop=config['eval_dataset']['loop'],
        )
        
        # 학습 데이터 로더
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=config['batch_size'],
            num_workers=config['data_loader']['num_workers'],
            drop_last=config['train_dataset']['loop'],
            collate_fn=IPADataCollator(processor),
        )
        # 평가 데이터 로더
        dataloader_eval = DataLoader(
            dataset_eval,
            batch_size=1,
            num_workers=0,
            drop_last=config['eval_dataset']['loop'],
            collate_fn=IPADataCollator(processor),
        )
        
        # wav2vec2 사전 학습 모델 불러오기
        model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base",
            vocab_size=len(tokenizer),
            ctc_loss_reduction="mean", 
            pad_token_id=processor.tokenizer.pad_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        
        # 학습 스케줄러 설정 (무한 루프 여부에 따라 다르게)
        if config['train_dataset']['loop']:
            config['scheduler_warmup_steps'] = len(dataset_train) * 2 // config['accumulation_steps'] // config['batch_size']
            config['scheduler_steps'] = config['total_iter'] // config['accumulation_steps']
        else:
            steps_per_epoch = math.ceil(len(dataloader_train) / config['accumulation_steps'])
            config['scheduler_warmup_steps'] = steps_per_epoch * 2
            config['scheduler_steps'] = steps_per_epoch * config['total_iter']
        
        # 옵티마이저 정의 (AdamW 사용)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['optimizer_vale']['lr'],
            weight_decay=config['optimizer_vale']['weight_decay'],
        )
        # 학습률 스케줄러 설정 (선형 warmup 및 decay)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['scheduler_warmup_steps'],
            num_training_steps=config['scheduler_steps']
        )
        
        # 평가자 클래스 정의
        evaluater = IPAModelEvaluator(
            dataloader=dataloader_eval,
            processor=processor,
            model=model,
            device=config['device'],
            loop=config['eval_dataset']['loop'],
        )
        
        # 학습자 클래스 정의
        trainer = IPAModelTrainer(
            dataloader=dataloader_train,
            model=model,
            optimizer=optimizer,
            processor=processor,
            tokenizer=tokenizer,
            lr_scheduler=lr_scheduler,
            evaluator=evaluater,
            loop=config['train_dataset']['loop'],
            **config['trainer'],
            **config,
        )
        
        # 학습 실행
        trainer.train()
    elif config['mode'] == 'test':
        # 여러 모델을 순회하며 평가
        for i in range(len(config['models'])):
            # 저장된 tokenizer, processor, model 불러오기
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(os.path.join(config['models'][i], 'tokenizer'))
            feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=False, return_attention_mask=False)
            processor = Wav2Vec2Processor.from_pretrained(os.path.join(config['models'][i], 'processor'))
            model = Wav2Vec2ForCTC.from_pretrained(os.path.join(config['models'][i], 'model'))
            
            # 평가용 데이터셋 및 로더 정의
            dataset_eval = StreamingDatasetQueue(
                data_path=config['eval_dataset']['path'],
                sound_path_prefix=config['eval_dataset']['audio_path'],
                processor=processor,
                x=config['eval_dataset']['x'],
                y=config['eval_dataset']['y'],
                max_queue_size=config['max_queue_size'],
                refill_threshold=config['refill_threshold'],
                chunk_size=config['chunk_size'],
                batch_size=1,
                notify_end=True,
                loop=config['eval_dataset']['loop'],
            )
            dataloader_eval = DataLoader(
                dataset_eval,
                batch_size=1,
                num_workers=0,
                drop_last=config['eval_dataset']['loop'],
                collate_fn=IPADataCollator(processor),
            )
            
            # 평가 클래스 정의
            evaluater = IPAModelEvaluator(
                dataloader=dataloader_eval,
                processor=processor,
                model=model,
                device=config['device'],
            )
            
            # 평가 실행 및 결과 출력
            logger.info(f"\nmodel: {config['models'][i]}")
            evaluater.evaluate_cer()
