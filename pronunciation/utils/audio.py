import torch
import torchaudio

import numpy as np
import librosa


def load_soundfile(audio_path: str) -> torch.Tensor:
    if audio_path.endswith('pcm'):
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
    return waveform
