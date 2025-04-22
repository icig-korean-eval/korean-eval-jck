from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer, get_linear_schedule_with_warmup

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

def evaluate_cer(model, processor, eval_csv_path):
    cer = load("cer")
    model.eval()
    device = "cpu"

    df = pd.read_pickle(eval_csv_path)

    predictions = []
    references = []
    
    print(processor.tokenizer.get_vocab())

    for file, ref_text, _, _ in tqdm(df.itertuples(index=False), desc='eval', total=df.shape[0]):
        waveform, sr = torchaudio.load(os.path.join('../data/announcer/source', file))
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        input_values = processor.feature_extractor(
            waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_values.to(device)

        with torch.no_grad():
            logits = model(torch.tensor(input_values, dtype=torch.float)).logits
        # print(logits.shape)
        pred_ids = torch.argmax(logits, dim=-1)[0]
        # print(pred_ids.shape)
        # print(pred_ids[:20])

        pred_text = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
        # print('pred:', processor.tokenizer.decode(pred_ids, skip_special_tokens=False), 'end')
        predictions.append(pred_text)
        references.append(ref_text)
        
        del file, ref_text
        
    p = []
    for idx, i in enumerate(zip(predictions, references)):
        if idx > 6: break
        p.append(f'{i[0]} || {i[1]}')
    
    print('\n\n'.join(p))
        
    score = cer.compute(predictions=predictions, references=references)
    print(f"CER (Character Error Rate): {score:.4f}")
    return score

if __name__ == "__main__":
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('./save/2025-04-22_10-50-02/iter_7200/tokenizer')
    processor = Wav2Vec2Processor.from_pretrained('./save/2025-04-22_10-50-02/iter_7200/processor')
    model = Wav2Vec2ForCTC.from_pretrained('./save/2025-04-22_10-50-02/iter_7200/model')
    evaluate_cer(model, processor, '../data/announcer/labeling/preprocessed_test.pickle')
