# Korean Speech-to-IPA Transcription Model

- A fine-tuned model that transcribes Korean audio speech into IPA (International Phonetic Alphabet)
  - Model: Fine-tuned [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
  - Dataset: Provided by AI Hub
    - Anchor speech version:  
      [News Script and Anchor Speech Data](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71557)
    - General speech version:  
      [Korean Speech](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=123)


## Project Overview

This project converts Korean spoken by non-native speakers into IPA symbols based on what is heard, and evaluates their pronunciation accuracy by comparing with the ground-truth IPA transcription. Two versions of the model were trained: one based on formal anchor speech and the other on standard general Korean speech.



## Training & Implementation

### Hyperparameter Management

- YAML files are used for efficient parameter configuration
  - Training parameters: [train.yaml](/pronunciation/config/train.yaml)
  - Testing parameters: [test.yaml](/pronunciation/config/test.yaml)

### Dataset

#### [StreamingDatasetQueue](./dataset/queue_dataset.py)

- The dataset size ranges from 25GB to 72GB, making it infeasible to load everything into memory.
- Implemented `StreamingDatasetQueue` by extending `IterableDataset` to stream data in chunks while training.

##### Implementation Principle

- A fixed-size queue (length `n`) loads indices 0 to n-1 initially.
- When the remaining queue size drops below a threshold, it loads the next chunk (n to 2n-1).
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

- Repeatedly yields data from the queue

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

- [vocab](/pronunciation/model/ipa_vocab_auto.json): Consists of 53 tokens including IPA characters and special tokens.

### Gradient Accumulation

- Applied to address memory limitations and enable a larger effective batch size.

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

- Applied to stabilize training by preventing exploding gradients.
  
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
  Directory containing `.yaml` files specifying **hyperparameters** used in training and testing.

- `dataset/`: 
  Package containing classes for **loading and preprocessing** audio datasets.

  - `collator.py`: 
    Implements a **DataCollator class** that preprocesses data into batches suitable for the model.

  - `queue_dataset.py`: 
    Implements a **queue-based streaming dataset class** that loads only a limited portion of data into memory.

- `logger/`: 
  Package implementing **logging utilities** for monitoring loss, accuracy, and other training statistics.

- `model/`: 
  Package containing **tokenizer vocabularies**, IPA-Hangul mapping tables, and preprocessing functions for model input.

- `trainer/`: 
  Package managing the training and evaluation processes.

  - `trainer.py`: 
    Implements a **Trainer class** that handles training loops, checkpointing, and loss tracking.

  - `evaluater.py`: 
    Implements an **Evaluater class** that evaluates the model's performance using test datasets.

- `utils.py`: 
  Defines **utility functions** for audio file loading and path handling used across the project.

- `main.py`: 
  The main **entry point script** that loads configuration and initiates training or testing.


## 결과 

| Model Version          | Evaluation Dataset | CER (Character Error Rate) |
|------------------------|--------------------|-----------------------------|
| Anchor Speech Version  | Anchor Speech      | **6.87%**                   |
| Anchor Speech Version  | General Speech     | **36%**                     |
| General Speech Version | General Speech     | **8.32%**                   |

- **Anchor Speech Version**: Suitable for evaluating formal, standardized pronunciation.  
- **General Speech Version**: More suitable for assessing natural speech closer to the Seoul dialect.
