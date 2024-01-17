from model import *
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
def contrast(net,val_loader,args):
    net.load_state_dict(torch.load("ckpt/climate/best.pth"))
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    error=[]
    with torch.no_grad():
        for (x,y) in tqdm(val_loader,desc="testing",leave=False):
            h0=torch.zeros((x.shape[0],args.hidden_size)).to(device)
            c0=torch.zeros((x.shape[0],args.hidden_size)).to(device)
            # x=x.transpose(1,0).to(device)
            output = net((x.to(device),y.to(device))).to(device)
            error0=output.clone().cpu().detach()-y
            error0=torch.flatten(error0)
            print(output)
            error.extend(error0.tolist())
    for i in range(output.shape[0]):
        tmp=output[i].cpu().detach()
        yi=y[i]
        plt.figure()
        plt.plot(tmp.tolist(), label='predicted value')
        plt.plot(yi.tolist(), label='real value')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(f'results/climate/contrast/contrast{i}.png')
        plt.close()
    return error

if __name__=="__main__":
    parser = argparse.ArgumentParser("lab4")
    parser.add_argument("--model", default="GRU")  # RNN GRU LSTM Bi-LSTM
    parser.add_argument("--epoches", type=int,default=30)
    parser.add_argument("--batch", type=int,default=128)  
    parser.add_argument("--task", default="climate")  # shopping or climate
    parser.add_argument("--output_size", type=int,default=10)
    parser.add_argument("--hidden_size", type=int,default=512)
    parser.add_argument("--input_size", default=256, type=int) 
    parser.add_argument("--learning_rate",type=float,default=1e-3,help="you know")
    parser.add_argument("--num_cls", default=10, type=int)

    args = parser.parse_args()
    train(args)
