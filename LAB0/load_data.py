import pandas as pd
import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset
import re
def clean(desstr,restr=' '):  
    #过滤表情   
    try:  
        co = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+')  
    except re.error:  
        co = re.compile(u'('u'\ud83c[\udf00-\udfff]|'u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'u'[\u2600-\u2B55])+')  
    return co.sub(restr, desstr)

def readcsv(filename,maxlen):
    data=pd.read_csv(filename,encoding="utf-8")

    data=shuffle(data)
    data=data if len(data)<maxlen else data[:maxlen]

    comments, labels = list(data[: ]['Reviews']), list(data[:]['Rating'])
    comments=[clean(str(i)) for i in comments]


    return comments,labels
def one_hot(y,num_cls):
    Y=torch.zeros(num_cls)
    Y[y-1]=1
    return Y
class CommentsDataset(Dataset):
    def __init__(self,comments,labels,split='train',pretrained_name='bert-base-cased',num_cls=5) -> None:
        super().__init__()
        self.split=split
        self.num_cls=num_cls
        assert len(comments)==len(labels)
        self.split_idx=int(0.7*len(comments))
        if self.split=='train':
            self.comments=comments[:self.split_idx]
            self.labels=labels[:self.split_idx]
        else:
            self.comments=comments[self.split_idx:]
            self.labels=labels[self.split_idx:]
    
    def __len__(self):
        return len(self.comments)
    def __getitem__(self, index):
        #X = self.tokenizer(self.comments[index], padding=True, truncation=True, return_tensors='pt')
        X=self.comments[index]
        Y=one_hot(self.labels[index],self.num_cls)
        return X,Y
