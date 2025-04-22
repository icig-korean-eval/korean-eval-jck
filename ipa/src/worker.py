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
    # supported transcription conventions: ipa, yale, park
    convention = convention.lower()
    if convention not in ['ipa', 'yale', 'park']:
        raise ValueError(f"Your input {convention} is not supported.")
    return ConversionTable(convention)


def sanitize(word: str) -> str:
    """
    converts all hanja 漢字 letters to hangul
    and also remove any space in the middle of the word
    """
    if len(word) < 1:  # if empty input, no sanitize
        return word

    # word = word.replace(' ', '')

    hanja_idx = [match.start() for match in re.finditer(r'\p{Han}', word)]
    if len(hanja_idx) == 0:  # if no hanja, no sanitize
        return word

    from src.hanja_tools import hanja_cleaner  # import hanja_cleaner only when needed
    r = hanja_cleaner(word, hanja_idx)
    return r


def convert(hangul: str,
            rules_to_apply: str = 'pastcnhovr',
            convention: str = 'ipa',
            sep: str = '',
            vebose=False) -> dict:
    # the main function for IPA conversion
    if vebose:
        print(f"original: {hangul}")

    if len(hangul) < 1:  # if no content, then return no content
        return ""
    
    original = hangul
    
    idx_to_str = dict()
    str_to_idx = []
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

    # prepare
    rules_to_apply = rules_to_apply.lower()
    CT_convention = transcription_convention(convention)
    hangul = sanitize(hangul)
    word = Word(hangul=hangul)
    if vebose:
        print(f'hangul: {hangul}')
        print(f'word jamo: {word.jamo}')
        print(f'word jamo idx: {word.jamo_idx}')
        print(f'word cv: {word.cv}')
        print()

    # resolve word-final consonant clusters right off the bat
    rules.simplify_coda(word)
    if vebose:
        print(f'simplify word jamo: {word.jamo}')
        print(f'simplify word jamo idx: {word.jamo_idx}')
        print(f'simplify word cv: {word.cv}')
        print()

    # apply rules
    word = rules.apply_rules(word, rules_to_apply)
    if vebose:
        print(f'rules word jamo: {word.jamo}')
        print(f'rules word cv: {word.cv}')
        print()

    # high mid/back vowel merger after bilabial (only for the Yale convention)
    if CT_convention.name == 'yale' and 'u' in rules_to_apply:
        bilabials = list("ㅂㅃㅍㅁ")
        applied = list(word.jamo)
        for i, jamo in enumerate(word.jamo[:-1]):
            if jamo in bilabials and word.jamo[i+1] == "ㅜ":
                applied[i+1] = "ㅡ"
        word.jamo = ''.join(applied)
    # print(f'CT_convention: {CT_convention}')
    # print()

    # convert to IPA or Yale
    transcribed = rules.transcribe(word.jamo, CT_convention)
    if vebose:
        print(f'transcribed: {transcribed}')
        print()
        print()

    # apply phonetic rules
    if CT_convention.name == 'ipa':
        transcribed = rules.apply_phonetics(transcribed, rules_to_apply)
    if vebose:
        print(f'transcribed ipa: {transcribed}')
        print()
        
    result = dict()
    result['original'] = original
    # result['result'] = sep.join(transcribed)
    
    idx = 0
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


def convert_many(long_content: str,
                 rules_to_apply: str = 'pastcnhovr',
                 convention: str = 'ipa',
                 sep: str = '') -> Union[int, str]:
    # decode uploaded file and create a wordlist to pass to convert()
    decoded = b64decode(long_content).decode('utf-8')
    decoded = decoded.replace('\r\n', '\n').replace('\r', '\n')  # normalize line endings
    decoded = decoded.replace('\n\n', '')  # remove empty line at the file end

    input_internal_sep = '\t' if '\t' in decoded else ','

    if '\n' in decoded:
        # a vertical wordlist uploaded
        input_lines = decoded.split('\n')
        wordlist = [l.split(input_internal_sep)[1].strip() for l in input_lines if len(l) > 0]
    else:
        # a horizontal wordlist uploaded
        wordlist = decoded.split(input_internal_sep)

    # iterate over wordlist and populate res
    res = ['Orthography\tIPA']
    for word in wordlist:
        converted_r = convert(hangul=word,
                              rules_to_apply=rules_to_apply,
                              convention=convention,
                              sep=sep)
        res.append(f'{word.strip()}\t{converted_r.strip()}')

    return '\n'.join(res)


if __name__ == "__main__":
    example = convert("예시")
    print(example)   # jɛ s i