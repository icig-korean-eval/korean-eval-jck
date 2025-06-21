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

import regex as re
import math

# 한글 유니코드 관련 상수 정의
GA_CODE = 44032 # 유니코드에서 '가'의 시작 코드 (한글 음절의 시작점)
G_CODE = 12593 # 유니코드에서 'ㄱ'의 시작 코드 (한글 자모의 시작점)
ONSET = 588 # 초성 간 유니코드 간격
CODA = 28 # 종성 간 유니코드 간격

# 초성 리스트 (인덱스 0 ~ 18)
ONSET_LIST = ('ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ')

# 중성 리스트 (인덱스 0 ~ 20)
VOWEL_LIST = ('ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ',
              'ㅡ', 'ㅢ', 'ㅣ')

# 종성 리스트 (인덱스 0 ~ 27, 0은 받침 없음)
CODA_LIST = ('', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ',
             'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ')


def hangul_to_jamos(hangul: str) -> list:
    """
    한글 문자열을 자모 단위로 분리하는 함수
    예: "안녕" → ["ㅇㅏㄴ", "ㄴㅕㅇ"]

    :param hangul: 변환할 한글 문자열
    :return: 음절별 자모 문자열이 담긴 리스트
    """
    
    # 한글 문자열을 문자 단위로 나눔
    syllables = list(hangul)
    r = []
    idx = []

    for letter in syllables:
        if bool(re.match(r'^[가-힣]+$', letter)): # 한글 음절(가~힣)에 해당하는 경우
            chr_code = ord(letter) - GA_CODE # '가'를 기준으로 몇 번째 한글인지 계산
            onset = math.floor(chr_code / ONSET) # 초성 인덱스 계산
            vowel = math.floor((chr_code - (ONSET * onset)) / CODA) # 중성 인덱스 계산
            coda = math.floor((chr_code - (ONSET * onset) - (CODA * vowel))) # 종성 인덱스 계산

            # 초성 + 중성 + 종성 문자열 생성
            syllable = f'{ONSET_LIST[onset]}{VOWEL_LIST[vowel]}{CODA_LIST[coda]}'
        else:
            # 한글이 아닌 경우 그대로 추가
            syllable = letter
        r.append(syllable)
    return r


def jamo_to_hangul(syllable: str) -> str:
    """
    자모 문자열을 한글 음절로 합성하는 함수 (단, 하나의 음절 단위로만 작동)
    예: "ㅇㅏㄴ" → "안"

    :param syllable: 자모 2~3자로 이루어진 문자열 (초성+중성+[종성])
    :return: 결합된 한글 음절 문자
    """
    if len(syllable) > 1:
        jamos = list(syllable)
        onset = ONSET_LIST.index(jamos[0]) # 초성 인덱스
        vowel = VOWEL_LIST.index(jamos[1]) # 중성 인덱스
        coda = CODA_LIST.index(jamos[2]) if len(syllable) == 3 else 0 # 종성 인덱스 (없으면 0)

        # 유니코드 공식 계산: (((초성 * 21) + 중성) * 28) + 종성 + GA_CODE
        utf_pointer = (((onset * 21) + vowel) * 28) + coda + GA_CODE
        syllable = chr(utf_pointer)
    return syllable


if __name__ == '__main__':
    pass