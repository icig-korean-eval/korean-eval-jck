from ipa.src.worker import convert

if __name__ == "__main__":
    original = [
        "넌 지금까지 어떤 학문을 공부했니? 난 아무것도 하지 않고 배고파서 닭고기 먹었닭.",
        "엄마는 맏이에게 특별한 선물을 주었다.", # 구개음화
        "역사 수업에서 북한의 문화에 대해 배웠다.", # 격음화
        "비가 없어서 행사가 취소되었다.", # 음운동화
        "나는 밖에서 친구를 만났다.", # 경음화 (표준발음법 제23항 적용)
        "저녁에 닭을 구워 먹었다.", # 자음군 단순화
        "어둠 속에 빛이 반짝였다.", # 음절말 장애음 중화
        "그는 좋은 음악을 들었다.", # 공명음 사이 ‘ㅎ’ 삭제
        "학생들이 책상을 정리했다.", # 공명음 사이 장애음 유성음화
    ]
    for o in original:
        result = convert(o, rules_to_apply='pastcnovr', sep=' ', vebose=False)
        print(f'original: {o}\nresult: {result['result']}')
        print()
        
        for k in result['words']:
            print(result['words'][k]['value'])
            print(' '.join(result['words'][k]['syllables']['jamo']))
            print(' '.join(result['words'][k]['syllables']['transcript']))
            print()
        
        print("=" * 50)
