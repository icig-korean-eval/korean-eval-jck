from ipa.src.worker import convert

if __name__ == "__main__":
    original = "넌 지금까지 어떤 학문을 공부했니? 난 아무것도 하지 않고 배고파서 밥 먹었어."
    example = convert(original, sep=' ')
    print(f'original: {original}\nresult: {example}')