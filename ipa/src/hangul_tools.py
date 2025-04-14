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

GA_CODE = 44032  # The unicode representation of the Korean syllabic orthography starts with GA_CODE
G_CODE = 12593   # The unicode representation of the Korean phonetic (jamo) orthography starts with G_CODE
ONSET = 588
CODA = 28

# ONSET LIST. 00 -- 18
ONSET_LIST = ('ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ')

# VOWEL LIST. 00 -- 20
VOWEL_LIST = ('ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ',
              'ㅡ', 'ㅢ', 'ㅣ')

# CODA LIST. 00 -- 27 + 1 (0 for open syllable)
CODA_LIST = ('', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ',
             'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ')


def hangul_to_jamos(hangul: str) -> list:
    # convert hangul strings to jamos
    # hangul: str. multiple korean letters like 가나다라마바사
    syllables = list(hangul)
    r = []
    idx = []

    for letter in syllables:
        if bool(re.match(r'^[가-힣]+$', letter)):     # if letter is a hangul character
            chr_code = ord(letter) - GA_CODE
            onset = math.floor(chr_code / ONSET)
            vowel = math.floor((chr_code - (ONSET * onset)) / CODA)
            coda = math.floor((chr_code - (ONSET * onset) - (CODA * vowel)))

            syllable = f'{ONSET_LIST[onset]}{VOWEL_LIST[vowel]}{CODA_LIST[coda]}'
        else:   # if letter is NOT a hangul character
            syllable = letter
        r.append(syllable)
    return r


def jamo_to_hangul(syllable: str) -> str:
    # only accept one syllable length of jamos and convert it to one hangul character
    if len(syllable) > 1:
        jamos = list(syllable)
        onset = ONSET_LIST.index(jamos[0])
        vowel = VOWEL_LIST.index(jamos[1])
        coda = CODA_LIST.index(jamos[2]) if len(syllable) == 3 else 0

        utf_pointer = (((onset * 21) + vowel) * 28) + coda + GA_CODE
        syllable = chr(utf_pointer)
    return syllable


if __name__ == '__main__':
    pass