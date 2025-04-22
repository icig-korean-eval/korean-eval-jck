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

from logger.logger import DefaultLogger


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train: 학습, test: 테스트')
    args = parser.parse_args()
    
    if args.mode == 'train':
        yaml_path = './config/train.yaml'
    elif args.mode == 'test':
        yaml_path = './config/test.yaml'
    else:
        raise argparse.ArgumentError(
            message=f'mode는 train, test만 가능합니다.'
        )
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    config['excute_time'] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config['save_path'] = os.path.join('./save', config['excute_time'])
    os.makedirs(config['save_path'], exist_ok=True)
    
    config['log']['path'] = config['save_path']
    
    with open(os.path.join(config['save_path'], 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f)
        
    config['mode'] = args.mode
    
    return config


if __name__ == "__main__":
    config = parse_config()
    logger = DefaultLogger(**config['log'])
    logger.info(f'condig: {config}')
    
    config['total_iter'] //= config['batch_size']
    config['accumulation_steps'] //= config['batch_size']
    config['print_steps'] //= config['batch_size']
    config['eval_steps'] //= config['batch_size']
    config['save_steps'] //= config['batch_size']
    
    tokenizer = Wav2Vec2CTCTokenizer(config['vocab_path'], unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" ")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=False, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    logger.info(f'tokens: {tokenizer.get_vocab()}')
    
    dataset = StreamingDatasetQueue(
        data_path=config['train_dataset']['path'],
        processor=processor,
        max_queue_size=config['max_queue_size'],
        refill_threshold=config['refill_threshold'],
        chunk_size=config['chunk_size'],
        batch_size=config['batch_size']
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['data_loader']['num_workers'],
        drop_last=config['data_loader']['drop_last'],
        collate_fn=IPADataCollator(processor),
    )
    
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        vocab_size=len(tokenizer),
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    config['device'] = device = "cuda" if torch.cuda.is_available() else "cpu"
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer_vale']['lr'],
        weight_decay=config['optimizer_vale']['weight_decay'],
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(dataset) * 2 // config['accumulation_steps'] // config['batch_size'],
        num_training_steps=config['total_iter'] // config['accumulation_steps']
    )
    
    if config['mode'] == 'train':
        evaluater = IPAModelEvaluator(
            processor=processor,
            model=model,
            device=config['device'],
            **config['eval_dataset'],
        )
        
        trainer = IPAModelTrainer(
            dataloader=dataloader,
            model=model,
            optimizer=optimizer,
            processor=processor,
            tokenizer=tokenizer,
            lr_scheduler=lr_scheduler,
            evaluator=evaluater,
            **config['trainer'],
            **config,
        )
        
        trainer.train()
        
        
