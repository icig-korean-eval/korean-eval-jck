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
    config['path'] = os.path.join('./save', config['excute_time'])
    os.makedirs(config['path'], exist_ok=True)
    
    config['log']['path'] = config['path']
    
    with open(os.path.join(config['path'], 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f)
    
    return config


if __name__ == "__main__":
    config = parse_config()
    logger = DefaultLogger(**config['log'])
    
    logger.info(f'condig: {config}')
