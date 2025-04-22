from torch.utils.data import Dataset
import torchaudio
import torch

class IPADataset(Dataset):
    def __init__(self, audio_paths, ipa_texts, processor):
        self.audio_paths = audio_paths
        self.ipa_texts = ipa_texts
        self.processor = processor

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        wav_path = self.audio_paths[idx]
        text = self.ipa_texts[idx]

        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        input_features = self.processor.feature_extractor(
            waveform.squeeze().numpy(), sampling_rate=16000
        )["input_features"][0]

        labels = self.processor.tokenizer(text, padding="max_length", max_length=128,
                                          return_tensors="pt").input_ids[0]

        return {
            "input_features": torch.tensor(input_features),
            "labels": labels
        }