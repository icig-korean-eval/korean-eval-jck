# -------------------------------------------------------
# This code is derived from code originally written by stannam (2022)
# and is distributed under the MIT License.
#
# Original Author: stannam
# Copyright (c) 2022 stannam
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -------------------------------------------------------

# phonological rules
import regex as re

from ipa.src.classes import Word, ConversionTable
from typing import Union


# 다양한 발음 규칙을 적용하기 위한 변환 테이블 로드
CT_double_codas = ConversionTable('double_coda')
CT_neutral = ConversionTable('neutralization')
CT_tensification = ConversionTable('tensification')
CT_assimilation = ConversionTable('assimilation')
CT_aspiration = ConversionTable('aspiration')
CT_convention = ConversionTable('ipa')

# 자음 리스트 (IPA 테이블의 C 열에서 특수기호(#, $) 제외)
CONSONANTS = tuple(list(CT_convention.C)[:-2])
# 모음 리스트
VOWELS = tuple(list(CT_convention.V))
# 공명음(sonorants) 자음: ㄴ, ㄹ, ㅇ, ㅁ
C_SONORANTS = ('ㄴ', 'ㄹ', 'ㅇ', 'ㅁ')
# 장애음(obstruents): 전체 자음에서 공명음을 뺀 나머지
OBSTRUENTS = tuple(set(CONSONANTS) - set(C_SONORANTS))
# 공명음 전체 = 모음 + 자음 공명음
SONORANTS = VOWELS + C_SONORANTS


def get_substring_ind(string: str, pattern: str) -> list:
    """
    문자열 내에서 특정 패턴이 등장하는 시작 인덱스들을 반환
    lookahead를 사용해 겹치는 패턴도 모두 탐색
    """
    return [match.start() for match in re.finditer(f'(?={pattern})', string)]


def transcribe(jamos: str, convention: ConversionTable = CT_convention, str_return: bool = False) -> Union[list, str]:
    """
    자모열(jamos)을 IPA 발음 기호로 전사하는 함수

    Parameters:
    - jamos: 자모열 문자열 (예: ㅁㅏㄴㄴㅕㅇ)
    - convention: ConversionTable 객체 (기본값은 ipa 테이블)
    - str_return: True일 경우 문자열로 반환, False일 경우 리스트 반환

    Returns:
    - 전사된 발음기호 문자열 또는 리스트
    """
    transcribed = []
    for jamo in jamos:
        is_C = convention.safe_index('C', jamo)  # 자음 여부
        is_V = convention.safe_index('V', jamo)  # 모음 여부
        if is_V >= 0:
            transcribed.append(convention.VSymbol[is_V])  # 모음 기호 추가
        elif is_C >= 0:
            transcribed.append(convention.CSymbol[is_C])  # 자음 기호 추가

    if str_return:
        return ''.join(transcribed)
    return transcribed


def palatalize(word: Word) -> str:
    """
    구개음화 적용 함수
    - 예: 받침 ㄷ/ㅌ + ㅣ → ㅈ/ㅊ 으로 변환

    Parameters:
    - word: Word 클래스 객체 (자모 정보를 포함)

    Returns:
    - 구개음화가 적용된 자모 문자열과 해당 자모의 인덱스 리스트
    """
    palatalization_table = {
        'ㄷ': 'ㅈ',
        'ㅌ': 'ㅊ'
    }
    # 한글 이외 문자 제거
    not_hangul = r'[^가-힣ㄱ-ㅎㅏ-ㅣ]'
    cleaned_hangul = re.sub(not_hangul, '', word.hangul) 
        
     # 음절 단위로 분리 후 자모로 변환
    hangul_syllables = list(cleaned_hangul)
    to_jamo_bound = word.to_jamo
    syllables_in_jamo = [to_jamo_bound(syl)[0] for syl in hangul_syllables]
    new_idx = []
    for i, syllable in enumerate(syllables_in_jamo):
        # print(syllable)
        try:
            next_syllable = syllables_in_jamo[i + 1]
            # 다음 음절이 'ㅣ'로 시작할 경우 구개음화 적용
            if next_syllable[0] == 'ㅣ':
                new_coda = palatalization_table.get(syllable[-1], syllable[-1])
                syllables_in_jamo[i] = ''.join(list(syllables_in_jamo[i])[:-1] + [new_coda])
        except IndexError:
            pass
        new_idx.extend([i + 1] * len(syllables_in_jamo[i]))
    new_jamo = ''.join(syllables_in_jamo)
    # print(new_idx)
    return new_jamo, new_idx


def aspirate(word: Word) -> str:
    """
    격음화 적용 함수
    - 예: ㅂ → ㅍ, ㄷ → ㅌ 등

    Parameters:
    - word: Word 객체

    Returns:
    - 격음화가 적용된 자모 문자열
    """
    return CT_aspiration.sub(word.jamo, word.jamo_idx)


def assimilate(word: Word) -> str:
    """
    자음 동화 적용 함수
    - 예: ㅂ + ㄴ → ㅁ + ㄴ

    Parameters:
    - word: Word 객체

    Returns:
    - 동화가 적용된 자모 문자열
    """
    return CT_assimilation.sub(word.jamo, word.jamo_idx)


def pot(word: Word) -> str:
    """
    경음화(된소리되기) 적용 함수
    - 예: ㅂ → ㅃ, ㄷ → ㄸ

    Parameters:
    - word: Word 객체

    Returns:
    - 경음화가 적용된 자모 문자열
    """
    return CT_tensification.sub(word.jamo, word.jamo_idx)


def neutralize(word: Word) -> str:
    """
    음절 말 자음 중화를 적용하는 함수.
    음절의 끝에 위치한 장애음을 'ㄷ' 등으로 중화하여 표준 발음에 맞게 변환.

    Parameters:
    - word: Word 클래스 객체

    Returns:
    - 중화가 적용된 자모 문자열
    """
    new_jamos = list(word.jamo)
    for i, jamo in enumerate(new_jamos):
        # 문장의 마지막이거나 다음 자모가 자음이면 중화 규칙 적용
        if i == len(new_jamos) - 1 or word.cv[i + 1] == 'C':
            new_jamos[i] = CT_neutral.apply(jamo)
    return ''.join(new_jamos)


def delete_h(word: Word) -> str:
    """
    사이시옷처럼 존재하는 'ㅎ' 삭제 규칙.
    공명음 사이에 있는 'ㅎ'은 삭제하는 것이 표준 발음 규칙임.

    Parameters:
    - word: Word 클래스 객체

    Returns:
    - 'ㅎ'이 삭제된 자모 문자열
    """
    h_locations = get_substring_ind(string=word.jamo, pattern='ㅎ')

    for h_location in reversed(h_locations):
        # 단어 처음이나 끝에 있는 'ㅎ'은 삭제하지 않음
        if h_location == 0 or h_location == len(word.jamo) - 1:
            # a word-initial h cannot undergo deletion
            continue
        preceding = word.jamo[h_location - 1]
        succeeding = word.jamo[h_location + 1]
        # 공명음 사이에 위치한 경우 삭제
        if preceding in SONORANTS and succeeding in SONORANTS:
            word.jamo = word.jamo[:h_location] + word.jamo[h_location + 1:]
            del word.jamo_idx[h_location]
    return word.jamo


def simplify_coda(input_word: Word, word_final: bool = False) -> Word:
    """
    복자음(자음군)을 단순화하는 규칙 적용
    - 예: ㄳ → ㄱ, ㄵ → ㄴ 등

    Parameters:
    - input_word: Word 클래스 객체

    Returns:
    - 복자음이 단순화된 Word 객체
    """
    def simplify(jamo: str, jamo_idx: list, loc: int) -> str:
        """
        자모열에서 특정 위치의 복자음을 단순화

        Parameters:
        - jamo: 자모 문자열
        - jamo_idx: 각 자모의 단어 내 인덱스 리스트
        - loc: 복자음 시작 위치

        Returns:
        - 단순화된 자모 문자열
        """
        list_jamo = list(jamo)
        
        before = ''.join(list_jamo[:loc + 1])
        double_coda = ''.join(list_jamo[loc + 1:loc + 3])
        after = ''.join(list_jamo[loc + 3:])

        converted = CT_double_codas.apply(text=double_coda, find_in='_separated')
        idx = len(double_coda) - len(converted)
        while idx > 0:
            del jamo_idx[loc + 1]
            idx -= 1
        return before + converted + after

    while True:
        # 음절 내부 복자음(VCCC) 위치 탐색
        double_coda_loc = get_substring_ind(input_word.cv, 'VCCC')
        if len(double_coda_loc) == 0:
            break
        # print(input_word.jamo[double_coda_loc[0]:double_coda_loc[0] + 4])
        # print(input_word.cv[double_coda_loc[0]:double_coda_loc[0] + 4])

        cc = double_coda_loc[0]  # work on the leftest CCC
        new_jamo = simplify(input_word.jamo, input_word.jamo_idx, cc)
        input_word.jamo = new_jamo

    # 단어 말미의 복자음(CC$)도 단순화
    final_CC = get_substring_ind(input_word.cv, 'CC$')
    if len(final_CC) > 0:
        cc = final_CC[0] - 1
        new_jamo = simplify(input_word.jamo, input_word.jamo_idx, cc)
        input_word.jamo = new_jamo
    return input_word


def non_coronalize(input_word: Word) -> str:
    """
    후설음화 규칙 (non-coronalization)
    비치경음 자음 앞에서 ㄴ → ㅇ, ㅁ으로 변화

    Parameters:
    - input_word: Word 클래스 객체

    Returns:
    - 후설음화 적용된 자모 문자열
    """
    velars = list('ㄱㅋㄲ') # 연구개음
    bilabials = list('ㅂㅍㅃㅁ') # 양순음
    non_velar_nasals = list('ㅁㄴ') # 비연구개 비음

    res = list(input_word.jamo)
    for i, jamo in enumerate(input_word.jamo[:-1]):
        if i == 0 or jamo not in non_velar_nasals:
            continue
        succeeding = input_word.jamo[i+1]
        if succeeding in velars:
            res[i] = 'ㅇ'
        elif succeeding in bilabials:
            res[i] = 'ㅁ'
    return ''.join(res)


def inter_v(symbols: list) -> list:
    """
    공명음 사이 무성음의 유성음화 처리 (intervocalic voicing)
    예: p → b, t → d, k → ɡ, tɕ → dʑ

    Parameters:
    - symbols: IPA 기호 리스트

    Returns:
    - 유성음화 적용된 기호 리스트
    """
    voicing_table = {
        'p': 'b',
        't': 'd',
        'k': 'ɡ',
        'tɕ': 'dʑ'
    }
    ipa_sonorants = [transcribe(s, str_return=True) for s in SONORANTS]

    res = list(symbols)

    for index, symbol in enumerate(symbols[:-1]):
        if index == 0 or symbol not in voicing_table.keys():
            continue
        preceding = symbols[index - 1]
        succeeding = symbols[index + 1]

        if preceding in ipa_sonorants:
            if succeeding in ipa_sonorants:
                res[index] = voicing_table.get(symbol, symbol)
            elif succeeding == 'ɕ':
                res[index] = voicing_table.get(symbol, symbol)
                res[index + 1] = 'ʑ'

    return res


def alternate_lr(symbols: list) -> list:
    """
    'ㄹ'의 위치에 따라 l / ɾ 변환
    - 모음 사이의 'ㄹ'은 'ɾ'로 전사 (flap)

    Parameters:
    - symbols: IPA 기호 리스트

    Returns:
    - 변환된 기호 리스트
    """
    ipa_vowels = [transcribe(v, str_return=True) for v in VOWELS]

    res = list(symbols)

    l_locs = [index for index, value in enumerate(symbols) if value == 'l']

    for l_loc in reversed(l_locs):
        if l_loc == 0 or l_loc == (len(symbols) - 1):
            continue

        preceding = symbols[l_loc - 1]
        succeeding = symbols[l_loc + 1]
        if preceding in ipa_vowels and succeeding in ipa_vowels:
            res[l_loc] = 'ɾ'

    return res


def apply_rules(word: Word, rules_to_apply: str = 'pastcnhovr') -> Word:
    """
    자모 수준에서 다양한 음운 규칙을 순차적으로 적용하는 함수.
    각 규칙의 앞글자를 통해 어떤 규칙을 적용할지 설정할 수 있음.

    Parameters:
    - word: Word 클래스 객체 (자모열, 인덱스 정보 포함)
    - rules_to_apply: 적용할 규칙을 문자열로 명시 (예: 'pastcnhovr')

    규칙 약어와 의미:
    - (p) Palatalization: 구개음화 ('ㄷ'/'ㅌ' + 'ㅣ' → 'ㅈ'/'ㅊ')
    - (a) Aspiration: 격음화 ('ㅎ'의 영향으로 격음 발생)
    - (s) Assimilation: 음운동화 (비음화, 유음화 등)
    - (t) Tensification: 경음화 (예외 없는 경음화 규칙, 표준발음법 23항)
    - (c) Complex coda simplification: 자음군 단순화
    - (n) Neutralization: 음절말 장애음 중화
    - (h) H-deletion: 공명음 사이 'ㅎ' 삭제
    - (o) Non-coronalization: 비치경음화 (ex. ㄴ → ㅇ)
    - (v) Voicing: 장애음의 유성음화 (공명음 사이)
    - (r) Liquids alternation: 'l' → 'ɾ' (모음 사이의 flap)

    Returns:
    - 규칙이 적용된 Word 객체
    """
    
    # print(1, word.jamo)
    # print(word.cv)
    
    # 1. 구개음화: ㄷ/ㅌ + ㅣ 조합이 있을 경우 적용
    if 'p' in rules_to_apply and ('ㄷㅣ' in word.jamo or 'ㅌㅣ' in word.jamo):
        word.jamo, word.jamo_idx = palatalize(word)
        word = simplify_coda(word) # 구개음화 후 자음군이 생길 수 있으므로 단순화

    # apply aspiration
    # print(2, word.jamo)
    # print(word.cv)
    
    # 2. 격음화: 'ㅎ'이 포함되어 있다면 적용
    if 'a' in rules_to_apply and 'ㅎ' in word.jamo:
        word.jamo = aspirate(word)

    # apply place assimilation
    # print(3, word.jamo)
    # print(word.cv)
    
    # 3. 음운동화: 동일 음소군 내에서의 변화 (비음화, 유음화 등)
    if 's' in rules_to_apply:
        word.jamo = assimilate(word)

    # apply post-obstruent tensification
    # print(4, word.jamo)
    # print(word.cv)
    
    # 4. 경음화: 장애음 뒤에 오는 예사소리-경음
    if 't' in rules_to_apply and any(jm in word.jamo for jm in OBSTRUENTS):
        word.jamo = pot(word)

    # apply complex coda simplification
    # print(5, word.jamo)
    # print(word.cv)
    
    # 5. 자음군 단순화: 복자음-단자음
    if 'c' in rules_to_apply:
        word = simplify_coda(word)

    # 6. 음절말 자음 중화: 예외없는 종성 변환
    if 'n' in rules_to_apply:
        word.jamo = neutralize(word)

    # 7. 공명음 사이의 'ㅎ' 삭제
    if 'h' in rules_to_apply and 'ㅎ' in word.jamo[1:-1]:
        word.jamo = delete_h(word)

    # 8. 후설음화: ㄴ/ㅁ이 ㄱ,ㅋ 등 뒤에 올 때 ㅇ,ㅁ으로 변화
    if 'o' in rules_to_apply:
        word.jamo = non_coronalize(word)

    return word


def apply_phonetics(ipa_symbols: list, rules_to_apply: str) -> list:
    """
    자모 기반 규칙 처리 이후, IPA 심벌 수준의 후처리 규칙 적용

    Parameters:
    - ipa_symbols: 자모를 변환한 IPA 기호 리스트
    - rules_to_apply: 'v', 'r' 등을 포함한 규칙 문자열

    규칙:
    - (v) Voicing: 공명음 사이 장애음 유성음화
    - (r) 'l' → 'ɾ' : 모음 사이에 위치한 'l'을 플랩(ɾ)으로 변환

    Returns:
    - 후처리 규칙이 적용된 IPA 기호 리스트
    """
    
    # 1. 장애음 유성음화: p → b, t → d 등
    if 'v' in rules_to_apply:
        ipa_symbols = inter_v(ipa_symbols)
        
    # 2. 'l' - 'ɾ': 모음 사이에서 l - ɾ 로 변환 (플랩)
    if 'r' in rules_to_apply and 'l' in ipa_symbols:
        ipa_symbols = alternate_lr(ipa_symbols)
    return ipa_symbols


if __name__ == '__main__':
    pass