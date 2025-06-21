import torch
import torchaudio

import numpy as np
import librosa


def load_soundfile(audio_path: str) -> torch.Tensor:
    # 입력 파일이 PCM 형식인 경우
    if audio_path.endswith('pcm'):
        audio_path = '.' + audio_path[2:]
        # PCM 파일을 바이너리 모드로 열기
        with open(audio_path, 'rb') as opened_pcm_file:
            buf = opened_pcm_file.read()
            # 16비트 정수형(int16)으로 변환하여 numpy 배열로 저장
            pcm_data = np.frombuffer(buf, dtype = 'int16')
            # librosa의 buf_to_float를 이용해 float32로 정규화된 waveform으로 변환
            waveform = librosa.util.buf_to_float(pcm_data)
            # PyTorch 텐서로 변환
            waveform = torch.tensor(waveform, dtype=torch.float32)
    else:
        # PCM이 아닌 일반 오디오 파일일 경우 torchaudio로 로딩
        waveform, sr = torchaudio.load(audio_path)
        # 샘플레이트가 16kHz가 아니면 16kHz로 리샘플링
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    return waveform
