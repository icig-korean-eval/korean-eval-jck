import pandas as pd
import json

# datas = pd.read_pickle('../data/announcer/labeling/preprocessed.pickle')
datas = pd.read_csv('./model/ipa2ko.csv')
print(datas.head())
character = set(datas['IPA'].values)
vocab = {c: i for i, c in enumerate(character)}
vocab['[UNK]'] = len(vocab)
vocab['[PAD]'] = len(vocab)

with open('./model/ipa_vocab_auto.json', mode='w', encoding='utf-8') as f:
    json.dump(vocab, f, indent=4)