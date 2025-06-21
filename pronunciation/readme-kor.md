# 한국어 발화 음성 IPA 변환 모델

- 한글 음성 파일을 입력 받아 IPA로 전사(transcription) 하는 모델 파인튜닝
  - 모델: [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) 파인튜닝
  - AI Hub에서 제공하는 데이터 활용
    - 아나운서 버전: 
    [뉴스 대본 및 앵커 음성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71557)
    - 일반 발화 버전: 
        [한국어 음성](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=123)


## 프로젝트 개요

외국인이 발음하는 한국어를 들리는 그대로 IPA로 전사하여 외국인의 발음을 측정하고, 원본 문장의 IPA와 비교하여 발음 정확도를 평가합니다. 아나운서 발음 기준으로 전사하는 모델과 일반적인 표준어 기준으로 전사하는 모델 두 버전으로 학습했습니다.


## 학습 및 구현

### 하이퍼파라미터 관리

- yaml 파일을 이용해 효율적으로 파라미터 관리
  - train 파라미터: [train.yaml](/pronunciation/config/train.yaml)
  - test 파라미터: [test.yaml](/pronunciation/config/test.yaml)

### 데이터셋

#### [StreamingDatasetQueue](./dataset/queue_dataset.py)

- 학습에 사용되는 데이터셋이 최소 25GB, 최대 72GB로 한번에 모든 데이터를 메모리에 로드하는것이 불가능했음
- 일정 개수의 데이터를 순차적으로 로드해서 제한된 메모리보다 더 큰 데이터셋을 학습할 수 있도록 `IterableDataset`를 상속하여 `StreamingDatasetQueue` 구현

##### 구현 원리

- 최대 길이 n의 Queue를 만들어 지정한 최대 개수 만큼 0~n-1 인덱스 먼저 로드
- 일정 개수 이하로 내려가면 n~2n-1 인덱스 로드
    ```python
    def _refill_queue(self):
        with self.lock:
            end = min(self.cursor + self.chunk_size, self.data_len)
            if self.cursor == end:
                return
            self.logger.debug(f'refill data: {self.cursor}\tapproximate remain: {self.q.qsize()}')
            for i in range(self.cursor, end):
                item = self._load_data(
                    file=os.path.join(self.sound_path_prefix, self.data_index[i][0]),
                    text=self.data_index[i][1]
                )
                if item is None:
                    self.logger.error(f'length error - idx: {i}')
                    continue
                self.q.put(item)
            if self.loop:
                if end < self.data_len:
                    self.cursor = end
                else:
                    self.cursor = 0
                    if self.notify_end:
                        self.q.put('end')
            else:
                self.cursor = end
    ```

- 일정 시간 간격으로 반복하며 남은 개수 체크

    ```python
    def __iter__(self):
        if self.loop:
            while True:
                item = self.q.get()
                yield item
                del item
        else:
            try:
                for _ in range(self.data_len):
                    item = self.q.get()
                    yield item
                    del item
            finally:
                self._reset()
    ```

###  Tokenizer

- [vocab](/pronunciation/model/ipa_vocab_auto.json): IPA 기호 + 특수토큰이 포함된 53개의 vocabulary 구성

### Gradient Accumulation

- vram이 부족해서 배치사이즈가 지나치게 작은 문제를 해결하기 위해 Gradient Accumulation 적용

    ```python
    def __update_model(self, input_values, attention_mask, labels, iter, last_batch=False, last_batch_section=False, remain=1):
        outputs = self.model(input_values=input_values, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        if not self.loop and last_batch_section:
            loss = loss / remain
        else:
            loss = loss / self.accumulation_steps
        loss.backward()
        
        if iter % self.accumulation_steps == 0 or (last_batch and last_batch_section):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.8)
            self.optimizer.step()
            # if self.loop:
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        return loss.item() / self.print_steps
    ```

### Gradient Clipping

- 학습 안정화를 위해 Gradient Clipping 적용
  
    ```python
    ...
    if iter % self.accumulation_steps == 0 or (last_batch and last_batch_section):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.8)
        ...
    ```


## 프로젝트 구조

```text
.
├── config/
├── dataset/
│   ├── collator.py
│   └── queue_dataset.py
├── logger/
│   └── ...
├── model/
│   └── ...
├── trainer/
│   ├── trainer.py
│   └── evaluater.py
├── utils.py
└── main.py
```

- `config/`:  
  학습 및 테스트 과정에서 사용할 **하이퍼파라미터를 설정한 `.yaml` 파일**들이 저장된 디렉토리입니다.

- `dataset/`:  
  음성 데이터셋 로딩 및 전처리를 위한 클래스들을 정의한 패키지입니다.

  - `collator.py`:  
    모델 입력에 맞게 배치 단위로 데이터를 전처리하는 **DataCollator 클래스**를 구현한 파일입니다.

  - `queue_dataset.py`:  
    **메모리 사용량을 제한하기 위해 큐(queue)**를 활용해 일정 크기의 데이터만 메모리에 유지하는 **데이터셋 클래스**를 구현한 파일입니다.

- `logger/`:  
  학습 중 손실 값, 정확도 등의 정보를 효과적으로 기록하고 모니터링하기 위한 **로깅 기능**을 구현한 패키지입니다.

- `model/`:  
  **토크나이저 vocabulary**, IPA-한글 변환 테이블 등 모델 학습에 필요한 **전처리 리소스** 및 처리 함수를 정의한 패키지입니다.

- `trainer/`:  
  학습 및 평가 과정 전체를 제어하는 Trainer 클래스를 구현한 패키지입니다.

  - `trainer.py`:  
    학습 루프, 체크포인트 저장, 손실 계산 등을 수행하는 **Trainer 클래스**를 정의한 파일입니다.

  - `evaluater.py`:  
    테스트 데이터셋을 이용해 모델 성능을 평가하는 **Evaluater 클래스**를 정의한 파일입니다.

- `utils.py`:  
  오디오 파일 로딩, 파일 경로 처리 등 학습 전반에서 사용되는 **보조 기능 함수들**을 정의한 파일입니다.

- `main.py`:  
  설정 파일을 불러와 학습 또는 테스트를 시작하는 **진입점(entry point)** 역할을 하는 실행 파일입니다.


## 결과 

| 모델 버전         | 평가 데이터      | CER (문자 오류율) |
|------------------|------------------|-------------------|
| 아나운서 발화 버전 | 아나운서 발화     | **6.87%**         |
| 아나운서 발화 버전 | 일반 발화         | **36%**         |
| 일반 발화 버전     | 일반 발화         | **8.32%**         |

- **아나운서 발화 버전**: 정확한 표준 발음을 학습하는 데 적합  
- **일반 발화 버전**: 서울 표준에 가까운 자연스러운 발음 평가에 적합
