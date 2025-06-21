import torch
from torch.nn.utils.rnn import pad_sequence


class IPADataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        # 만약 batch가 단순 문자열 리스트이면 그대로 반환
        # 평가 과정에서 transcription 결과 문자열만 전달될 수 있음
        if isinstance(batch[0], str):
            return batch

        # batch는 여러 개의 데이터(dict)로 구성되어 있음
        # 각 항목에서 input_values, labels, script를 추출
        input_values = [item["input_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        scripts = [item["script"] for item in batch]

        # 길이 정보 저장 (나중에 마스크 생성이나 길이 손실 계산에 사용)
        input_lengths = [len(x) for x in input_values]
        label_lengths = [len(x) for x in labels]
        
        # input_values를 가장 긴 길이에 맞게 padding (padding_value로 패딩)
        input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=self.processor.feature_extractor.padding_value)
        try:
            # labels도 패딩 (-100으로 패딩하여 loss에서 무시되도록 설정)
            labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        except:
            # 라벨 패딩에 문제가 생기면 라벨을 출력하고 종료
            print(labels)
            exit()

        # attention_mask 생성: 실제 값 있는 부분은 1, 나머지는 0
        attention_mask = torch.zeros_like(input_values_padded).long()
        for i, l in enumerate(input_lengths):
            attention_mask[i, :l] = 1

        # 최종 결과 반환: 모델 입력에 필요한 모든 텐서 포함
        return {
            "input_values": input_values_padded,             # 음성 입력 (패딩 포함)
            "attention_mask": attention_mask,                # 마스크
            "labels": labels_padded,                         # 타겟 레이블
            "scripts": scripts,                              # 문자열 텍스트
            "input_lengths": torch.tensor(input_lengths),    # 실제 입력 길이
            "label_lengths": torch.tensor(label_lengths)     # 실제 라벨 길이
        }