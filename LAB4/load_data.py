from torch.utils.data import Dataset,DataLoader
import torch
import pandas as pd
import random
import argparse
import math
from utils import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
shopping_cats = {'书籍':0, '平板':1, '手机':2, '水果':3, '洗发水':4, '热水器':5, '蒙牛':6, '衣服':7, '计算机':8, '酒店':9}
class ShoppingDataset(Dataset):
    def __init__(self,data,cats,vocab):
        self.data=data
        self.vocab=vocab
        self.cats=cats
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index],shopping_cats[self.cats[index]]
    @staticmethod
    def collect_fn(batch):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        review_repr=[]
        cats=[]
        tmp,_=zip(*batch)
        max_len=len(max(tmp,key=len))
        for i in batch:
            r,c=i
            if len(r)<max_len:
                padding_list=['<pad>']*(max_len-len(r))
                r=r+padding_list
            r=map_vocab(r,config.vocab)
            review_repr.append(torch.LongTensor(r))
            cats.append(c)
        cats=torch.LongTensor(cats)
        review_repr=torch.stack(review_repr).transpose(1,0)
        return review_repr.to(device),cats.to(device)
    
def build_shopping():
    seg_reviews,cats=load_text()
    train_review=[]
    train_cats=[]
    test_review=[]
    test_cats=[]
    for i,seg in enumerate(seg_reviews):
        if (i+1)%5==0:
            test_review.append(seg)
            test_cats.append(cats[i])
        else:
            train_review.append(seg)
            train_cats.append(cats[i])
    train_data=list(zip(train_review,train_cats))
    random.shuffle(train_data)
    train_review,train_cats=zip(*train_data)
    vocab=build_vocab(train_review)
    return ShoppingDataset(train_review,train_cats,vocab),ShoppingDataset(test_review,test_cats,vocab),vocab


class ClimateDataset(Dataset):
    def __init__(self, data):
        super(ClimateDataset, self).__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        df = self.data[item]
        trainData = df.loc[df['WeekDay'] <= 4]
        resultData = df.loc[df['WeekDay'] >= 5]

        date = list(resultData['Date Time'])
        trainData = trainData.drop(['Date Time'], axis=1)
        resultData = resultData['T (degC)']

        trainData = trainData.values
        resultData = resultData.values

        return torch.Tensor(trainData),torch.Tensor(resultData),


    
def getWeatherData(config):
    trainData = pd.read_csv('dataset/train.csv')
    testData = pd.read_csv("dataset/test.csv")
    trainData = [data[1] for data in list(trainData.groupby('WeekIndex')) if len(data[1]) == 1008]
    testData = [data[1] for data in list(testData.groupby('WeekIndex')) if len(data[1]) == 1008]
    trainDataset = ClimateDataset(trainData)
    testDataset = ClimateDataset(testData)
    trainDataLoader = DataLoader(dataset=trainDataset, batch_size=config.batch, shuffle=True)
    testDataLoader = DataLoader(dataset=testDataset, batch_size=config.batch, shuffle=False)
    return (trainDataLoader, testDataLoader)



if __name__=="__main__":
    parser = argparse.ArgumentParser("lab4")
    parser.add_argument("--input_size", type=int,default=100)
    args = parser.parse_args()
    d=ClimateDataset()
    train_loader=DataLoader(d,batch_size=32,shuffle=False)
    for (x,y) in train_loader:
        print(x.transpose(1,0)[0])
        break
    # data=pd.read_csv('dataset/online_shopping_10_cats.csv',engine='python',encoding='utf-8')
    # print(data)