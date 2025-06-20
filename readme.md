# 외국인 한국어 발음 평가를 위한 IPA 변환 시스템

- 한글 문자열을 **국제 발음 기호(IPA)** 로 변환하는 기능
  - [stannam/hangul_to_ipa](https://github.com/stannam/hangul_to_ipa?tab=MIT-1-ov-file) 코드 변형
- 한글 음성 파일을 입력 받아 **IPA로 전사(transcription)** 하는 모델 파인튜닝
  - 모델: [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) 파인튜닝
  - AI Hub에서 제공하는 데이터 활용
    - 아나운서 버전: 
    [뉴스 대본 및 앵커 음성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71557)
    - 일반 발화 버전: 
        [한국어 음성](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=123)

## 프로젝트 개요

외국인이 발음하는 한국인의 발음 정확도를 평가하기 위한 기준으로 외국인이 얼마나 한국인의 발음을 잘 알아듣게 따라했는지 평가하기로 정했습니다. 평가 방법은 원본 문장의 발음 기호와, 사용자가 말한 음성 파일을 들리는대로 발음기호로 변환하여 얼마나 일치하는지 비교하는 방식입니다. 그래서 한국어 문자열이 주어지면 여러 발음 법칙을 적용해서 발음 기호로 변환하는 기능과, 한국어 발화 음성파일을 발음기호로 변환하는 모델을 만들었습니다.


## IPA 변환기

- 한글 음성파일 IPA 기호 변환 모델 학습과 데일리 러닝을 위한 원본 문장 IPA 변환을 위해 한글 문자열이 주어지면 여러 발음 법칙을 적용한 IPA 변환 기능을 구현
- [stannam/hangul_to_ipa](https://github.com/stannam/hangul_to_ipa)을 이용하여 구현

### 기능 추가

- 기존 프로젝트는 전사된 발음 기호가 음소 단위로 한 글자씩 출력되어 어떤 단어가 발음되는지 한눈에 알아보기 어려웠음.
- 위 문제를 해결하기 위해 단어 단위로 발음 기호가 출력되도록 변경
- 기능 추가 전 출력
  - 입력: 나는 음성인식이 재밌어요 
  - 출력: [n ɑ n ɯ n ɯ m s ʌ ŋ i n s i ɡ i dʑ ɛ m i s* ʌ jo]
- 기능 추가 후 출력
  - 입력: 나는 음성인식이 재밌어요 
  - 출력: nɑnɯn ɯmsʌŋinsiɡi dʑɛmis*ʌjo


## 한국어 음성파일 IPA 전사 모델

외국인이 말하는 발음을 한국인 기준에서 평가하기 위해 외국인의 발음을 한국인 기준으로 학습한 모델로 들리는대로 IPA로 전사하는 모델

### 데이터셋

#### 뉴스 대본 및 앵커 음성 데이터

- 한국인의 아나운서 발음을 기준으로 IPA 발음기호 전사 모델 학습을 위해 사용
- [데이터 다운로드](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71557)

##### 데이터 분석

- 전체 분량: 약 **1100시간** 음성-전사 텍스트 페어
- 전체 용량: 약 **230GB**
- 제한된 GPU 자원으로 인해 전체 데이터셋을 학습에 사용할 수는 없었으며, Validation으로 제공된 약 25GB 분량의 데이터만 학습에 활용
- 기존 데이터셋에 존재하던 transcribe 텍스트를 IPA 변환기를 통해 IPA 데이터셋 구축
- 최종적으로 한글 음성파일-IPA 페어 데이터셋 구축

#### 한국어 음성

- 일반적인 한국인의 표준어 발음을 기준으로 IPA 발음기호 전사 모델 학습을 위해 사용
- [데이터 다운로드](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=123)

##### 데이터 분석

- 전체 분량: 약 **1000시간** 음성-전사 텍스트 페어
- 전체 용량: 약 **72GB**
- 모든 데이터셋을 이용하여 학습
  - 데이터셋 대부분이 표준어를 사용하므로 별도의 선별 없이 모든 데이터셋을 활용 ([데이터 분석](https://www.mdpi.com/2076-3417/10/19/6936))
- 기존 데이터셋에 존재하던 transcribe 텍스트를 IPA 변환기를 통해 IPA 데이터셋 구축
- 최종적으로 한글 음성파일-IPA 페어 데이터셋 구축


### 학습 방식

- 아나운서 버전, 일반적인 한국어 버전 모두 동일한 프로세스로 학습 진행
- 자세한 학습 방식은 [pronunciation/readme.md](/pronunciation/readme.md) 참고


## 프로젝트 구조

```text
├── ipa/
    ├── ...
├── pronunciation/
    ├── ...
├── pronunciation-back/
    ├── ...
├── analyze_annoncer_data.ipynb
├── analyze_korean_data.ipynb
├── fine_tune_wav2vec2_for_english_asr_notebook.ipynb
└── ipa.py
```

- `ipa/`: 한글 문자열을 **국제 발음 기호(IPA)**로 변환하는 기능을 구현한 패키지입니다.  
  ➤ 자세한 내용은 [`ipa/readme.md`](ipa/readme.md) 참고

- `pronunciation/`: 
  한국어 음성 파일을 IPA 기호로 **전사하는 Wav2Vec2 기반 모델**을 학습하는 패키지입니다.  
  ➤ 자세한 내용은 [`pronunciation/readme.md`](pronunciation/readme.md) 참고

- `pronunciation-back/`: 
  `pronunciation`의 구버전입니다. 최종 제출에는 사용되지 않아 설명은 생략합니다.

- `analyze_annoncer_data.ipynb`: 
  AI Hub의 **아나운서 음성 데이터셋**을 분석하고, 학습용으로 구성하는 Jupyter Notebook입니다.

- `analyze_korean_data.ipynb`: 
  AI Hub의 **일반 한국어 음성 데이터셋**을 분석하고, 학습용으로 구성하는 Jupyter Notebook입니다.

- `fine_tune_wav2vec2_for_english_asr_notebook.ipynb`: 
  Wav2Vec2 모델을 파인튜닝하는 방법을 예시로 정리한 노트북입니다. *(설명 생략)*

- `ipa.py`: 
  IPA 변환 알고리즘을 실험적으로 수정하여 테스트하는 코드입니다.



## 최종 성능 결과

### 성능

| 모델 버전         | 평가 데이터      | CER (문자 오류율) |
|------------------|------------------|-------------------|
| 아나운서 발화 버전 | 아나운서 발화     | **0.0687**         |
| 아나운서 발화 버전 | 일반 발화         | **0.3610**         |
| 일반 발화 버전     | 일반 발화         | **0.0832**         |

- **아나운서 발화 버전**: 정확한 표준 발음을 학습하는 데 적합  
- **일반 발화 버전**: 서울 표준에 가까운 자연스러운 발음 평가에 적합

### 모델

- [아나운서 발화 버전](https://huggingface.co/icig/announcer-korean-ipa-translation)
- [일반 발화 버전](https://huggingface.co/icig/normal-korean-ipa-translation)


## License

This project is licensed under the Apache License 2.0.

We use the pretrained [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) model hosted on HuggingFace, which is also licensed under Apache 2.0.

For details, see [NOTICE](/NOTICE).
