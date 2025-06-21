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

# the engine that does the hard lifting.
# convert() is the entry point for converting Korean orthography into transcription

import regex as re
from base64 import b64decode
from typing import Union

from ipa.src.classes import ConversionTable, Word
import ipa.src.rules as rules


def transcription_convention(convention: str):
    """
    변환 방식(IPA, Yale, Park 등)을 받아 해당하는 ConversionTable을 반환.
    지원되는 변환 형식: 'ipa', 'yale', 'park'
    """
    convention = convention.lower()
    if convention not in ['ipa', 'yale', 'park']:
        raise ValueError(f"Your input {convention} is not supported.")
    return ConversionTable(convention)


def sanitize(word: str) -> str:
    """
    입력 문자열 내의 한자(漢字)를 한글로 변환하고,
    단어 중간의 공백을 제거하는 전처리 함수.
    - 단어가 비어있으면 그대로 반환
    - 한자가 없으면 그대로 반환
    """
    if len(word) < 1:
        return word

    # word = word.replace(' ', '')

    # 한자 위치 찾기 (\p{Han}은 유니코드 한자 블록)
    hanja_idx = [match.start() for match in re.finditer(r'\p{Han}', word)]
    if len(hanja_idx) == 0:  # if no hanja, no sanitize
        return word

    from src.hanja_tools import hanja_cleaner
    r = hanja_cleaner(word, hanja_idx)
    return r


def convert(hangul: str,
            rules_to_apply: str = 'pastcnhovr',
            convention: str = 'ipa',
            sep: str = '',
            vebose=False) -> dict:
    """
    입력된 한글 문자열을 IPA 또는 Yale 등으로 변환하는 핵심 함수

    Parameters:
    - hangul: 변환할 한글 문자열
    - rules_to_apply: 적용할 발음 규칙 문자열 (예: 'pastcnhovr')
    - convention: 변환 규약 ('ipa', 'yale', 'park')
    - sep: 출력 결과에서 음소를 구분할 구분자
    - vebose: 중간 디버깅 출력 여부

    Returns:
    - result: 변환 결과를 담은 딕셔너리
        {
            'original': 원문 문자열,
            'result': 전체 변환 결과,
            'result_array': 변환된 음소 배열,
            'words': 각 단어별 자모 및 음성기호 정보
        }
    """
    
    if vebose:
        print(f"original: {hangul}")

    if len(hangul) < 1:  # if no content, then return no content
        return ""
    
    # 원본 문자열 보존
    original = hangul
    
    # 문자열 내 각 문자의 위치 기록
    idx_to_str = dict() # 위치 - 문자
    str_to_idx = [] # 문자 - 위치
    idx = 1
    for letter in hangul:
        if bool(re.match(r'^[가-힣]+$', letter)):
            str_to_idx.append(idx)
            idx_to_str[idx] = letter
            idx += 1
        else:
            str_to_idx.append(letter)
    if vebose:
        print(f'idx_to_str: {idx_to_str}')
        print(f'str_to_idx: {str_to_idx}')

    # 설정 초기화
    rules_to_apply = rules_to_apply.lower()
    CT_convention = transcription_convention(convention)
    
    # 전처리 (한자 - 한글 변환 등)
    hangul = sanitize(hangul)
    
    # Word 객체 생성 (자모 분리 등)
    word = Word(hangul=hangul)
    if vebose:
        print(f'hangul: {hangul}')
        print(f'word jamo: {word.jamo}')
        print(f'word jamo idx: {word.jamo_idx}')
        print(f'word cv: {word.cv}')
        print()

    # 시작 단계에서 자음군 단순화 처리
    rules.simplify_coda(word)
    if vebose:
        print(f'simplify word jamo: {word.jamo}')
        print(f'simplify word jamo idx: {word.jamo_idx}')
        print(f'simplify word cv: {word.cv}')
        print()

    # 발음 규칙 적용
    word = rules.apply_rules(word, rules_to_apply)
    if vebose:
        print(f'rules word jamo: {word.jamo}')
        print(f'rules word cv: {word.cv}')
        print()

    # Yale 표기법일 경우, bilabial 뒤의 ㅜ - ㅡ로 교체하는 규칙 추가
    if CT_convention.name == 'yale' and 'u' in rules_to_apply:
        bilabials = list("ㅂㅃㅍㅁ")
        applied = list(word.jamo)
        for i, jamo in enumerate(word.jamo[:-1]):
            if jamo in bilabials and word.jamo[i+1] == "ㅜ":
                applied[i+1] = "ㅡ"
        word.jamo = ''.join(applied)
    # print(f'CT_convention: {CT_convention}')
    # print()

    # 자모를 음성기호(IPA 등)로 변환
    transcribed = rules.transcribe(word.jamo, CT_convention)
    if vebose:
        print(f'transcribed: {transcribed}')
        print()
        print()

    # IPA의 경우, 음성학적 규칙 추가 적용 (유성음화 등)
    if CT_convention.name == 'ipa':
        transcribed = rules.apply_phonetics(transcribed, rules_to_apply)
    if vebose:
        print(f'transcribed ipa: {transcribed}')
        print()
        
    # 결과 구성
    result = dict()
    result['original'] = original
    # result['result'] = sep.join(transcribed)
    
    idx = 0
    # 각 단어별로 자모 및 변환 결과 정리
    result['words'] = dict()
    for k, v in idx_to_str.items():
        result['words'][k] = dict()
        result['words'][k]['value'] = v
        result['words'][k]['syllables'] = dict()
        result['words'][k]['syllables']['jamo'] = []
        result['words'][k]['syllables']['transcript'] = []
        while idx < len(word.jamo_idx) and word.jamo_idx[idx] == k:
            result['words'][k]['syllables']['jamo'].append(word.jamo[idx])
            result['words'][k]['syllables']['transcript'].append(transcribed[idx])
            idx += 1
    
    # 전체 문장 기준 변환 결과 조립
    result['result'] = ''
    result['result_array'] = []
    for s in str_to_idx:
        if isinstance(s, int):
            result['result'] += ''.join(result['words'][s]['syllables']['transcript'])
            result['result_array'].extend(result['words'][s]['syllables']['transcript'])
        else:
            if s == ' ':
                result['result'] += s
                result['result_array'].append(s)

    return result


if __name__ == "__main__":
    example = convert("예시")
    print(example)   # jɛ s i