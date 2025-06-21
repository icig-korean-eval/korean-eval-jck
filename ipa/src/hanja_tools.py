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

import csv
from ipa.src.hangul_tools import hangul_to_jamos, jamo_to_hangul
from pathlib import Path

HIGHV_DIPHTHONGS = ("ㅑ", "ㅕ", "ㅖ", "ㅛ", "ㅠ", "ㅣ")  # 고모음 또는 반모음


def realize_hanja(raw: str) -> str:
    """
    한자의 유니코드 코드 포인트(U+XXXX 형태)를 실제 한자 문자로 변환하는 함수
    예: 'U+349A' → '㒚'
    """
    stripped_raw = raw.strip('U+')  # 앞의 'U+' 문자열 제거
    r = chr(int(stripped_raw, 16))  # 16진수 문자열을 정수로, 다시 문자로 변환
    return r


def load_jajeon() -> dict:
    """
    한자-한글 독음 매핑 정보를 담은 사전(자전)을 불러오는 함수.
    'tables/hanja.tsv' 파일을 기반으로 사전을 만듦.
    """
    jajeon = {}
    jajeon_path = Path(__file__).parent.parent / 'tables' / 'hanja.tsv'
    with open(jajeon_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            # the original file uses the Unicode code point (e.g., U+349A), so need to convert this to the actual hanja
            key = realize_hanja(row[0])
            value = row[1]
            jajeon[key] = value
    return jajeon


def hanja_to_hangul(jajeon: dict, char:str) -> str:
    """
    단일 한자 문자(char)를 자전(jajeon)을 사용해 한글로 변환.
    매핑이 없으면 원래 문자를 그대로 반환.
    """
    try:
        r = jajeon[char]
    except KeyError:
        r = char
    return r


def initial_rule(char: str) -> str:
    """
    두음법칙을 적용하여 초성 'ㄹ' → 'ㄴ', 또는 'ㄴ' → 'ㅇ'로 바꾸는 함수
    예: "렬" → "녈", "녀" → "여"
    """
    changed_flag = False
    jamos = hangul_to_jamos(char)
    jamos = ''.join(jamos)
    onset, nucleus = jamos[0], jamos[1]
    if onset == 'ㄹ':
        onset = 'ㄴ'
        changed_flag = True
    if onset == 'ㄴ' and nucleus in HIGHV_DIPHTHONGS:
        onset = 'ㅇ'
        changed_flag = True

    if changed_flag:
        jamo_list = list(jamos)
        jamo_list[0], jamo_list[1] = onset, nucleus
        jamos = ''.join(jamo_list)

    return jamo_to_hangul(jamos)


def hanja_cleaner(word: str, hanja_loc:list[int]) -> str:
    """
    문자열에서 특정 위치의 한자들을 한글로 변환하고,
    필요시 두음법칙이나 특수 발음 규칙도 함께 적용하는 함수

    :param word: 한자 및 한글이 섞인 문자열
    :param hanja_loc: 한자가 포함된 위치 리스트 (예: [0,2])
    :return: 한자 변환 및 규칙 적용된 한글 문자열
    """
    jajeon = load_jajeon()
    chars = list(word)

    for i in hanja_loc:
        if chars[i] in ["不", "不"] and (i < len(chars) - 1):  # if 不 appears in a non-ultimate syllable
            if chars[i + 1] == "實":
                # special case: 不實 = 부실
                chars[i] = "부"
                chars[i + 1] = "실"
                continue
            else:
                # special case: 不 is pronounced as 부[pu] before an alveolar ㄷㅈ
                chars[i + 1] = hanja_to_hangul(jajeon, chars[i + 1])
                next_syllable = hangul_to_jamos(chars[i + 1])
                following_onset = ''.join(next_syllable)[0]
                chars[i] = "부" if following_onset in ["ㄷ", "ㅈ"] else "불"
                continue

        chars[i] = hanja_to_hangul(jajeon, chars[i])

        if i == 0:  # apply the 'initial rule' (두음법칙)
            chars[i] = initial_rule(chars[i])

    return ''.join(chars)


if __name__ == '__main__':
    pass