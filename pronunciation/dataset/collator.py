import torch
from torch.nn.utils.rnn import pad_sequence


class IPADataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        input_values = [item["input_values"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_lengths = [len(x) for x in input_values]
        label_lengths = [len(x) for x in labels]
        
        input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=self.processor.feature_extractor.padding_value)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        attention_mask = torch.zeros_like(input_values_padded).long()
        for i, l in enumerate(input_lengths):
            attention_mask[i, :l] = 1

        return {
            "input_values": input_values_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "input_lengths": torch.tensor(input_lengths),
            "label_lengths": torch.tensor(label_lengths)
        }