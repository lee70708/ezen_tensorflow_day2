from konlpy.tag import Okt
import re
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# import nltk
# nltk.download()

okt = Okt()
ctx = 'C:/Users/ezen/PycharmProjects/test2/data/'
filename = ctx + 'kr-Report_2018.txt'

with open(filename, 'r', encoding='UTF-8') as f:
    texts = f.read()

texts = texts.replace('\n', '')
tokenizer = re.compile('[^ ㄱ-힣]+')
texts = tokenizer.sub('', texts)
tokens = word_tokenize(texts)

print(texts[:7])

noun_token = []
for token in tokens:
    token_pos = okt.pos(token)
    temp = [txt_tag[0] for txt_tag in token_pos if txt_tag[1] =="Noun"]
    if len(''.join(temp)) > 1:
        noun_token.append("".join(temp))
texts = " ".join(noun_token)
print(texts[:300])

with open(ctx+'stopwords.txt', 'r', encoding='UTF-8') as f:
    stopwords = f.read()

stopwords = stopwords.split(' ')
print(texts[:10])

texts = [text for text in tokens if text not in stopwords]
freqtxt = pd.Series(dict(FreqDist(texts))).sort_values(ascending=False)
print(freqtxt[:30])

okt.pos('가치창출')
okt.pos('갤러시')

wcloud = WordCloud(ctx + 'D2Coding.ttf', relative_scaling=0.2, background_color='white').generate(" ".join(texts))

plt.figure(figsize=(12,12))
plt.imshow(wcloud, interpolation='bilinear')
plt.axis("off")
plt.show()