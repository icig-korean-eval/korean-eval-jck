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
        # 'tables' 폴더 안의 {name}.csv 파일을 열어 탭 구분자로 읽음
        table_path = Path(__file__).parent.parent / 'tables' / f'{self.name}.csv'
        with open(table_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=',')
            # CSV의 각 행을 반복하면서 데이터를 읽음
            for row in reader:
                for header, value in row.items():
                    # 각 열(header)에 해당하는 값을 리스트에 저장
                    if not hasattr(self, header):
                        setattr(self, header, [])
                    getattr(self, header).append(value)
        # 모든 리스트를 튜플로 변환하여 불변성을 유지
        for header in reader.fieldnames:
            setattr(self, header, tuple(getattr(self, header)))

    def apply(self, text: str, find_in: str = '_from') -> str:
        """
        단일 문자 혹은 문자열을 변환 규칙에 따라 변경
        :param text: 변환 대상 텍스트 (문자열)
        :param find_in: 변환 기준으로 사용할 속성 이름 (예: '_from')
        :return: 변환된 텍스트 (문자열)
        """
        try:
            from_tuple = getattr(self, find_in)
            ind = from_tuple.index(text)
            return self._to[ind]
        except (AttributeError, ValueError):
            return text

    def sub(self, text: str, idx_list: List[int], find_in: str = '_from') -> Tuple[str, List[int]]:
        """
        주어진 텍스트 내의 서브스트링을 변환하고, 그에 따른 인덱스 리스트(idx_list)를 동기화
        :param text: 변환 대상 전체 문자열
        :param idx_list: 각 문자에 해당하는 단어 인덱스 리스트
        :param find_in: 기준이 되는 속성 (ex: '_from')
        :return: 변환된 텍스트와 인덱스 리스트
        """
        
        from_tuple = getattr(self, find_in)
        for index, item in enumerate(from_tuple):
            # 정규표현식을 이용해 item이 나타나는 모든 위치를 찾음
            positions = [match.start() for match in re.finditer(re.escape(item), text)]
            if len(positions) == 0: continue
            
            # 변환 규칙 적용: item → self._to[index]
            text = text.replace(item, self._to[index])
            
            # 인덱스 리스트(idx_list)도 텍스트 변환 길이에 맞게 보정
            for i in range(len(positions) - 1, -1, -1):
                idx = len(item) - len(self._to[index])
                while idx > 0:
                    del idx_list[positions[i]]
                    idx -= 1
        return text

    def safe_index(self, attribute, element):
        """
        지정된 속성(attribute)에서 element의 인덱스를 안전하게 반환
        :return: 인덱스 값 또는 없을 경우 -1
        """
        target_tuple = getattr(self, attribute)
        try:
            return target_tuple.index(element)
        except ValueError:
            return -1

    def __str__(self):
        return str(f'ConversionTable {self.name}')


class Word:
    def __init__(self, hangul):
        # 변환할 원래 단어 (한글)
        self.hangul = hangul
        # 단어를 자모로 변환하고, 각 자모가 어떤 단어에 속하는지를 저장
        self._jamo, self._jamo_idx = self.to_jamo(hangul)
        # 자모열에서 자음(C)과 모음(V)을 태깅하여 구조 분석 (예: CVCCV 등)
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
         # 자모가 바뀌면 CV 마킹도 다시 계산해야 하므로 함께 업데이트
        self._cv = self.mark_CV(self._jamo)

    @jamo_idx.setter
    def jamo_idx(self, value):
        self._jamo_idx = value

    @property
    def cv(self):
        return self._cv
    
    # def word_idx(self, hangul: str) -> str:

    def mark_CV(self, jamo: str, convention: ConversionTable = None) -> str:
        """
        자모 문자열에서 각 문자가 자음인지 모음인지 분석하여 'C' 또는 'V'로 태깅
        :param jamo: 자모 문자열
        :param convention: 자음/모음 기준으로 사용할 변환 테이블 (기본은 'ipa')
        :return: CV 구조 문자열 (예: 'CVCVC')
        """
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
        """
        한글 문자열을 자모열로 변환하며, 옵션에 따라 무음 초성 'ㅇ'을 제거하거나, 음절 경계를 처리함
        :param hangul: 한글 단어
        :param no_empty_onset: 초성이 'ㅇ'일 경우 제거할지 여부 (기본값 True)
        :param sboundary: 아직 구현되지 않은 음절 경계 처리 옵션
        :return: 자모 문자열과 각 자모가 어떤 단어 인덱스에 속하는지를 나타내는 리스트
        """
        # 한글 이외 문자는 제거
        not_hangul = r'[^가-힣ㄱ-ㅎㅏ-ㅣ]'
        cleaned_hangul = re.sub(not_hangul, '', hangul)
        
        # 한글을 자모열로 변환 (예: "안녕" → [["ㅇ", "ㅏ", "ㄴ"], ["ㄴ", "ㅕ", "ㅇ"]])
        jamo_forms = hangul_to_jamos(cleaned_hangul)

        # 복합 종성(예: ㄳ, ㄵ 등)을 두 개의 자음으로 분리
        jamo_forms = self.separate_double_coda(jamo_forms)

        # 초성이 무음(ㅇ)인 경우 해당 자모 제거
        if no_empty_onset:
            jamo_forms = self.remove_empty_onset(jamo_forms)

        if sboundary:
            # not implemented
            pass
        
        # 각 자모가 어떤 음절(단어) 인덱스에 속하는지 리스트로 저장
        jamo_idx = []
        for j in range(1, len(jamo_forms) + 1):
            jamo_idx.extend([j] * len(jamo_forms[j - 1]))
        return ''.join(jamo_forms), jamo_idx

    def remove_empty_onset(self, syllables: list[str]) -> list:
        """
        초성이 무음(ㅇ)일 경우 해당 자모 제거
        :param syllables: 자모 3개로 구성된 음절 리스트
        :return: 초성 'ㅇ'이 제거된 자모 리스트
        """
        r = []
        for syllable in syllables:
            to_append = syllable[1:] if syllable[0] == 'ㅇ' else syllable
            r.append(to_append)
        return r

    def separate_double_coda(self, syllables: list[str]) -> list:
        """
        복합 종성(ㄳ, ㄵ 등)을 두 자음으로 분리
        :param syllables: 자모 3개로 구성된 음절 리스트
        :return: 종성이 2개로 분리된 음절 리스트
        """
        r = []
        # double_coda 변환 규칙 테이블 로딩
        CT_double_codas = ConversionTable('double_coda')
        for syllable in syllables:
            # 종성이 없으면 그대로 저장
            if len(syllable) < 3:
                r.append(syllable)
                continue
            coda = syllable[2]
            try:
                # 종성이 double_coda에 해당하면 분리된 자모로 치환
                separated_coda = CT_double_codas._separated[CT_double_codas._double.index(coda)]
                r.append(syllable[:2] + separated_coda)
                continue
            except ValueError:
                r.append(syllable)
                continue
        return r

    def __str__(self):
        return self.hangul