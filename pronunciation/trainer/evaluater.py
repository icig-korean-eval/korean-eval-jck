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


class IPAModelEvaluator:
    def __init__(self,
                 processor: transformers.processing_utils.ProcessorMixin,
                 model: torch.nn.Module,
                 device: str = 'cpu',
                 path: str = None,
                 audio_path: str = None,):
        self.logger = DefaultLogger()
        self.processor = processor
        self.model = model
        self.device = device
        self.eval_dataset = self.__read_eval_data(path)
        self.eval_audio_path = audio_path
        
        self.model = self.model.to(device)
        
        
    def __read_eval_data(self, eval_data_path):
        if eval_data_path is None: return
        return pd.read_pickle(eval_data_path)
    
    
    def evaluate_cer(self):
        cer = load("cer")
        self.model.eval()

        predictions = []
        references = []

        for file, ref_text, _, _ in tqdm(
            self.eval_dataset.itertuples(index=False),
            desc='eval', total=self.eval_dataset.shape[0]
        ):
            audio_path = os.path.join(self.eval_audio_path, file)
            if file.endswith('pcm'):
                audio_path = '.' + audio_path[2:]
                with open(audio_path, 'rb') as opened_pcm_file:
                    buf = opened_pcm_file.read()
                    pcm_data = np.frombuffer(buf, dtype = 'int16')
                    waveform = librosa.util.buf_to_float(pcm_data)
                    waveform = torch.tensor(waveform, dtype=torch.float32)
            else:
                waveform, sr = torchaudio.load(audio_path)
                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

            input_values = self.processor.feature_extractor(
                waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
            ).input_values.to(self.device)

            with torch.no_grad():
                logits = self.model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)[0]

            pred_text = self.processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
            predictions.append(pred_text)
            references.append(ref_text)
            
            del file, ref_text
        
        score = cer.compute(predictions=predictions, references=references)
        self.logger.info(f"CER (Character Error Rate): {score:.4f}")
