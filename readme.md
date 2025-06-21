# IPA Conversion System for Evaluating Foreigners' Korean Pronunciation

- Functionality to convert Korean text into **International Phonetic Alphabet (IPA)**
  - Based on modified code from [stannam/hangul_to_ipa](https://github.com/stannam/hangul_to_ipa?tab=MIT-1-ov-file)
- Fine-tuning a model that takes Korean audio files and **transcribes them into IPA**
  - Model: Fine-tuned [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
  - Data from AI Hub:
    - Announcer version:  
      [News Script and Anchor Speech Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71557)
    - General speech version:  
      [Korean Speech](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=123)

## Project Overview

To evaluate how accurately a foreigner can pronounce Korean words, we assess how intelligible their pronunciation is to native speakers. The evaluation compares the IPA transcription of the original sentence with the IPA transcription derived from the foreign speaker’s audio. Therefore, we implemented a system that can convert Korean text to IPA using various pronunciation rules, and another system that transcribes Korean speech to IPA.


## IPA Converter

- Converts Korean text to IPA using pronunciation rules for use in model training and during daily learning
- Implemented based on [stannam/hangul_to_ipa](https://github.com/stannam/hangul_to_ipa)

### Feature Improvements


- The original project output IPA as individual phonemes, making it hard to identify which words were pronounced
- We modified the output to show IPA by word unit

- **Before modification:**
  - Input: 나는 음성인식이 재밌어요  
  - Output: `[n ɑ n ɯ n ɯ m s ʌ ŋ i n s i ɡ i dʑ ɛ m i s* ʌ jo]`

- **After modification:**
  - Input: 나는 음성인식이 재밌어요  
  - Output: `nɑnɯn ɯmsʌŋinsiɡi dʑɛmis*ʌjo`


## IPA Transcription Model for Korean Audio

A model that transcribes foreign-accented Korean speech into IPA based on native Korean phonological norms.

### Dataset

#### News Script and Anchor Speech Dataset

- Used to train the IPA transcription model based on professional announcer pronunciation
- [Download Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71557)

##### Dataset Analysis

- Total duration: approx. **1100 hours** of audio-text pairs
- Total size: approx. **230 GB**
- Due to limited GPU resources, only the validation portion (~25 GB) was used for training
- The original transcript text was converted into IPA using the IPA converter
- Final dataset: audio-IPA paired dataset

#### Korean Speech

- Used to train a model based on standard Korean pronunciation
- [Download Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=123)


##### Dataset Analysis

- Total duration: approx. **1000 hours** of audio-text pairs
- Total size: approx. **72 GB**
- All data used without filtering, as most speakers use standard Korean ([Data Source](https://www.mdpi.com/2076-3417/10/19/6936))
- Transcripts were converted into IPA using the IPA converter
- Final dataset: audio-IPA paired dataset


### Training Method

- Both the announcer and general speech versions were trained using the same process
- For details, see [pronunciation/readme.md](/pronunciation/readme.md)


## Project Structure

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

- `ipa/`: A package that implements **Korean text to IPA conversion**  
  ➤ See [`ipa/readme.md`](ipa/readme.md) for details

- `pronunciation/`: 
  A package for training a **Wav2Vec2-based model that transcribes Korean audio into IPA**  
  ➤ See [`pronunciation/readme.md`](pronunciation/readme.md) for details

- `pronunciation-back/`: 
  Legacy version of `pronunciation`. Not used in the final submission.

- `analyze_annoncer_data.ipynb`: 
  Jupyter Notebook for analyzing and preparing the **Announcer Speech Dataset** from AI Hub

- `analyze_korean_data.ipynb`: 
  Jupyter Notebook for analyzing and preparing the **Korean Speech Dataset** from AI Hub

- `fine_tune_wav2vec2_for_english_asr_notebook.ipynb`: 
  A sample notebook demonstrating how to fine-tune Wav2Vec2 *(description omitted)*

- `ipa.py`: 
  Code for experimenting with IPA conversion algorithm modifications



## Final Performance Results

### Accuracy

| Model Version       | Evaluation Data     | CER (Character Error Rate) |
|---------------------|---------------------|-----------------------------|
| Announcer Version   | Announcer Speech    | **6.87%**                   |
| Announcer Version   | General Speech      | **36%**                     |
| General Version     | General Speech      | **8.32%**                   |

- **Announcer version**: Suitable for evaluating pronunciation accuracy against standard professional speech
- **General version**: Suitable for evaluating natural speech closer to standard Seoul dialect

### Models

- [Announcer Speech Version](https://huggingface.co/icig/announcer-korean-ipa-translation)
- [General Speech Version](https://huggingface.co/icig/normal-korean-ipa-translation)


## Contribution

- Joonchul Kim - 100%  
  - Completed all work


## License

This project is licensed under the Apache License 2.0.

We use the pretrained [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) model hosted on HuggingFace, which is also licensed under Apache 2.0.

For details, see [NOTICE](/NOTICE).
