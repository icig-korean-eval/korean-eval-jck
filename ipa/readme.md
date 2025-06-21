# Hangul-to-IPA Converter

- Original project: [stannam/hangul_to_ipa](https://github.com/stannam/hangul_to_ipa)
- The original implementation generated phonetic transcriptions as sequences of individual phonemes, making it difficult to identify word boundaries.  
- This project modifies the output to display phonemes grouped by word for better readability.


## Project Structure

```text
src/
├── classes.py
├── hangul_tools.py
├── hanja_tools.py
└── rules.py
tables/
```

- `src/`: 
  Core directory containing classes and utility functions for decomposing Hangul and applying pronunciation rules.

  - `classes.py`: 
    Contains classes for splitting Korean words into graphemes and converting them into IPA using rule-based logic.

  - `hangul_tools.py`: 
    Utility functions for converting between Hangul syllables and individual graphemes (jamo).

  - `hanja_tools.py`: 
    Functions to handle Hanja (Chinese characters) within Korean strings.  
    *Note: These were not used in this project as no Hanja is present in the dataset.*

  - `rules.py`: 
    Defines phonological rules (e.g., nasal assimilation, fortition) and applies them to grapheme sequences.

- `tables/`: 
  A set of `.csv` mapping files used to apply phonological rules and match graphemes to IPA symbols.  
  Examples include: jamo-to-IPA mappings, liaison rule tables, etc.


## Key Modifications

### [src/worker.py](./src/worker.py)

- **convert()**
  - Added character-level index mapping to maintain word grouping.
  - Restores phoneme sequence based on word-character index dictionary.

### [src/classes.py](./src/classes.py)

- **ConversionTable.sub()**
  - Updated to allow Hangul-to-IPA substitution at the character level.

- **Word.to_jamo()**
  - Tracks which graphemes belong to which word index to enable grouped phoneme output.


## Result

### Pronunciation Rules Applied

- Palatalization (구개음화) 
- Aspiration (격음화) 
- Standard Pronunciation Rule 23 (unconditional fortition) (표준발음법 제23항(예외없는 경음화))
- Consonant cluster simplification (자음군단순화)
- Syllable-final obstruent neutralization (음절말 장애음 중화)
- Voicing of obstruents between sonorants (공명음 사이 장애음 유성음화)

### Example Output

- **Before modification**  
  Input: 나는 음성인식이 재밌어요  
  Output: 
  `[n ɑ n ɯ n ɯ m s ʌ ŋ i n s i ɡ i dʑ ɛ m i s* ʌ jo]`

- **After modification**  
  Input: 나는 음성인식이 재밌어요  
  Output: 
  `nɑnɯn ɯmsʌŋinsiɡi dʑɛmis*ʌjo`
