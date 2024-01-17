import scipy.io as scio
import numpy as np
from torch.utils.data import Dataset
import torch
data_dict=scio.loadmat('GAN/points.mat')

data=data_dict['xx']
np.random.shuffle(data)
class PointsDataset(Dataset):
    def __init__(self,data):
        super(PointsDataset,self).__init__()
        self.data=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float()
    
if __name__=="__main__":
    k=['a','b','c','d','xx']
    for i in k:
        print(data_dict[i].shape)