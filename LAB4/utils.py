import pandas as pd
from pkuseg import pkuseg
from load_data import *
import numpy as np
import time
import datetime
from torch.utils.data import Dataset,DataLoader
import torch
with open('stop-words/hit_stopwords.txt',encoding='utf-8') as f:
    lines=f.readlines()
stopwords={i.strip() for i in lines}
class config:
    vocab=None

def word_seg(text,cut):
    words=cut(text)
    filted_words=[]
    for word in words:
        if not word in stopwords:
            filted_words.append(word)
    return filted_words

def load_text(file='dataset/online_shopping_10_cats.csv'):
    seg=pkuseg(postag=False)
    df=pd.read_csv(file,encoding='utf-8')
    cats=df['cat'].tolist()
    reviews=df['review'].tolist()
    for i,text in enumerate(reviews):
        if type(text)!=str:
            reviews[i]=" "

    seg_reviews=[word_seg(r,seg.cut) for r in reviews]
    return seg_reviews,cats

def build_vocab(texts):
    vocab=set()
    for text in texts:
        vocab.update(text)
    vocab_dict = {word: idx+1 for idx, word in enumerate(vocab)}
    vocab_dict['unknown'] = 0
    vocab_dict['<pad>']=len(vocab_dict)
    return vocab_dict

def map_vocab(words,vocab):
    word_repr=[]
    for word in words:
        if vocab.get(word,None):
            word_repr.append(vocab[word])
        else:
            word_repr.append(vocab['unknown'])
    return word_repr



def weekday4train(row):
    time_cur = row['Date Time']
    time_end = datetime.datetime.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    time_start = datetime.datetime.strptime("01.01.2009 00:10:00", "%d.%m.%Y %H:%M:%S")
    return int((time_end - time_start).days % 7)

def weekday4test(row):
    time_cur = row['Date Time']
    time_end = datetime.datetime.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    time_start = datetime.datetime.strptime("01.01.2015 00:10:00", "%d.%m.%Y %H:%M:%S")
    return int((time_end - time_start).days % 7)

def week4train(row):
    time_cur = row['Date Time']
    time_end = datetime.datetime.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    time_start = datetime.datetime.strptime("01.01.2009 00:10:00", "%d.%m.%Y %H:%M:%S")
    return int((time_end - time_start).days / 7)

def week4test(row):
    time_cur = row['Date Time']
    time_end = datetime.datetime.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    time_start = datetime.datetime.strptime("01.01.2015 00:10:00", "%d.%m.%Y %H:%M:%S")
    return int((time_end - time_start).days / 7)

def sin4hour(row):
    time_cur = row['Date Time']
    time_val = time.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    return float(np.sin(time_val.tm_hour*(2*np.pi/24)))

def cos4hour(row):
    time_cur = row['Date Time']
    time_val = time.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    return float(np.cos(time_val.tm_hour*(2*np.pi/24)))

def sin4month(row):
    time_cur = row['Date Time']
    time_val = time.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    mon = time_val.tm_mon
    mon_sin = np.sin(mon*(2*np.pi/12))
    return float(mon_sin)

def cos4month(row):
    time_cur = row['Date Time']
    time_val = time.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    mon = time_val.tm_mon
    mon_cos = np.cos(mon*(2*np.pi/12))
    return float(mon_cos)

def year(row):
    time_cur = row['Date Time']
    time_val = time.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    return int(time_val.tm_year)

if __name__ == '__main__':

    weatherData = pd.read_csv("dataset/jena_climate_2009_2016.csv")
    weatherData['sin4hour'] = weatherData.apply(sin4hour, axis=1)
    weatherData['cos4hour'] = weatherData.apply(cos4hour, axis=1)
    weatherData['sin4month'] = weatherData.apply(sin4month, axis=1)
    weatherData['cos4month'] = weatherData.apply(cos4month, axis=1)
    weatherData['Year'] = weatherData.apply(year, axis=1)
    rowIndex1 = weatherData[(weatherData['Year'] <= 2014)].index.tolist()
    rowIndex2 = weatherData[(weatherData['Year'] >= 2015)].index.tolist()
    rowIndex1.append(rowIndex1[-1] + 1)
    trainData = weatherData.iloc[rowIndex1]
    rowIndex2 = rowIndex2[1:]
    testData = weatherData.iloc[rowIndex2]
    trainData['WeekIndex'] = trainData.apply(week4train, axis=1)
    testData['WeekIndex'] = testData.apply(week4test, axis=1)
    trainData['WeekDay'] = trainData.apply(weekday4train, axis=1)
    testData['WeekDay'] = testData.apply(weekday4test, axis=1)
    trainData.to_csv("dataset/train.csv", index=False)
    testData.to_csv("dataset/test.csv", index=False)


