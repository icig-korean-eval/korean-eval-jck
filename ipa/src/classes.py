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
import regex as re
from ipa.src.hangul_tools import hangul_to_jamos
from pathlib import Path

from typing import Tuple, List


class ConversionTable:
    def __init__(self, name):
        self.name = name
        # Open the tab-delimited file located in the 'tables' folder
        table_path = Path(__file__).parent.parent / 'tables' / f'{self.name}.csv'
        with open(table_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=',')
            # Iterate over each row in the file
            for row in reader:
                # For each header, set it as an attribute if it's not already set
                for header, value in row.items():
                    # Add the value to a list associated with the header
                    if not hasattr(self, header):
                        setattr(self, header, [])
                    getattr(self, header).append(value)
        # Convert lists to tuples because the contents should be immutable
        for header in reader.fieldnames:
            setattr(self, header, tuple(getattr(self, header)))

    def apply(self, text: str, find_in: str = '_from') -> str:
        # for a single phoneme, find it among _from (or any attribute name find_in)
        # and convert it to _to
        try:
            from_tuple = getattr(self, find_in)
            ind = from_tuple.index(text)
            return self._to[ind]
        except (AttributeError, ValueError):
            return text

    def sub(self, text: str, idx_list: List[int], find_in: str = '_from') -> Tuple[str, List[int]]:
        from_tuple = getattr(self, find_in)
        for index, item in enumerate(from_tuple):
            positions = [match.start() for match in re.finditer(re.escape(item), text)]
            if len(positions) == 0: continue
            text = text.replace(item, self._to[index])
            for i in range(len(positions) - 1, -1, -1):
                idx = len(item) - len(self._to[index])
                while idx > 0:
                    del idx_list[positions[i]]
                    idx -= 1
        return text

    def safe_index(self, attribute, element):
        target_tuple = getattr(self, attribute)
        try:
            return target_tuple.index(element)
        except ValueError:
            return -1

    def __str__(self):
        return str(f'ConversionTable {self.name}')


class Word:
    def __init__(self, hangul):
        # word to convert
        self.hangul = hangul
        self._jamo, self._jamo_idx = self.to_jamo(hangul)
        self._cv = self.mark_CV(self.jamo)

    @property
    def jamo(self):
        return self._jamo

    @property
    def jamo_idx(self):
        return self._jamo_idx

    @jamo.setter
    def jamo(self, value):
        self._jamo = value
        self._cv = self.mark_CV(self._jamo)

    @jamo_idx.setter
    def jamo_idx(self, value):
        self._jamo_idx = value

    @property
    def cv(self):
        return self._cv
    
    # def word_idx(self, hangul: str) -> str:

    def mark_CV(self, jamo: str, convention: ConversionTable = None) -> str:
        # identify each element in jamo as either consonant or vowel
        r = ''

        if convention is None:
            convention = ConversionTable('ipa')

        consonants = convention.C
        vowels = convention.V

        for j in jamo:
            if j in vowels:
                r += 'V'
            elif j in consonants:
                r += 'C'
        return r

    def to_jamo(self, hangul: str, no_empty_onset: bool = True, sboundary: bool = False) -> str:
        # Convert Hangul forms to jamo, remove empty onset ㅇ
        # e.g., input "안녕" output "ㅏㄴㄴㅕㅇ"
        not_hangul = r'[^가-힣ㄱ-ㅎㅏ-ㅣ]'
        cleaned_hangul = re.sub(not_hangul, '', hangul)  # hangul without special characters
        jamo_forms = hangul_to_jamos(cleaned_hangul)

        jamo_forms = self.separate_double_coda(jamo_forms)  # divide double coda (e.g., "ㄳ" -> "ㄱㅅ")

        if no_empty_onset:  # remove soundless syllable initial ㅇ
            jamo_forms = self.remove_empty_onset(jamo_forms)

        if sboundary:
            # not implemented
            pass
        
        jamo_idx = []
        for j in range(1, len(jamo_forms) + 1):
            jamo_idx.extend([j] * len(jamo_forms[j - 1]))
        return ''.join(jamo_forms), jamo_idx

    def remove_empty_onset(self, syllables: list[str]) -> list:
        r = []
        for syllable in syllables:
            to_append = syllable[1:] if syllable[0] == 'ㅇ' else syllable
            r.append(to_append)
        return r

    def separate_double_coda(self, syllables: list[str]) -> list:
        r = []
        CT_double_codas = ConversionTable('double_coda')
        for syllable in syllables:
            if len(syllable) < 3:
                r.append(syllable)
                continue
            coda = syllable[2]
            try:
                separated_coda = CT_double_codas._separated[CT_double_codas._double.index(coda)]
                r.append(syllable[:2] + separated_coda)
                continue
            except ValueError:
                r.append(syllable)
                continue
        return r

    def __str__(self):
        return self.hangul