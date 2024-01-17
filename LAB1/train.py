from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn, optim
from MLP import MLP
import os
import argparse
from logger import *
from tqdm import tqdm,trange


def onehot(target,num_cls):
    l=[]
    for y in target:
        tmp=torch.zeros(num_cls)
        tmp[y.item()]=1
        tmp=tmp.unsqueeze(0)
        l.append(tmp)
    return torch.cat(l,dim=0)

def train(args):
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((args.mean,),(args.std,))])
 
    #下载数据集
    os.makedirs('data',exist_ok=True)
    data_train = datasets.MNIST(root = "data",
                                transform=transform,
                                train = True,
                                download = True)
    
    data_test = datasets.MNIST(root="data",
                            transform = transform,
                            train = False,
                            download = True)
    #装载数据
    train_loader = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size = args.batch,
                                                    shuffle = True)
    
    test_loader = torch.utils.data.DataLoader(dataset=data_test,
                                                batch_size = args.batch,
                                                shuffle = True)

    net=MLP(args)
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)  
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []
    acc_list=[]

    net.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    net.to(device)
    max_acc=0
    os.makedirs('ckpt',exist_ok=True)
    logger = get_logger(args.logger)
    logger.info('start training!')
    cri=nn.CrossEntropyLoss()
    for epoch in trange(args.epoches,desc="training"):
        loss_epoch=0
        for batch_idx, (data, target) in enumerate(train_loader):
            #print(data.shape)
            output=net(data.to(device))
            optimizer.zero_grad()
            target_onehot=onehot(target,args.num_cls)
            loss=cri(output,target_onehot.to(device))
            loss.backward()
            optimizer.step()
            loss_epoch+=loss.item()
        
        train_losses.append(loss_epoch/len(train_loader.dataset))
        train_counter.append(epoch+1)

        if (epoch+1)%10==0:
            acc,test_loss=test(net,test_loader)
            acc_list.append(acc)
            test_losses.append(test_loss)
            test_counter.append(epoch)
            torch.save(net.state_dict(), f'ckpt/epoch{epoch}.pth')
            logger.info("epoch:{}/{}: train loss:{:.3f} test loss:{:.3f} acc:{:.3f}".format(epoch+1,args.epoches,loss_epoch/len(train_loader.dataset),test_loss,acc))
            if acc>max_acc:
                max_acc=acc
                torch.save(net.state_dict(), 'ckpt/best.pth')
                logger.info("better results!")
    visualization(train_counter,train_losses,test_counter,test_losses,acc_list,args)

def test(net,test_loader):

    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    test_loss = 0

    sum=0
    cri=nn.CrossEntropyLoss()
    for data, target in tqdm(test_loader,desc="testing",leave=False):
        output = net(data.to(device))
        target_onehot=onehot(target,args.num_cls)
        test_loss += cri(output,target_onehot.to(device)).item()
        sum+=(output.argmax(dim=1)==target.to(device)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc=sum/len(test_loader.dataset)

    return acc,test_loss
def visualization(train_counter,train_losses,test_counter,test_losses,acc_list,args):
    plt.figure()
    plt.title(f"lr:{args.learning_rate} momentum:{args.momentum} epoches:{args.epoches}")
    plt.plot(train_counter, train_losses, color='blue')
    plt.plot(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('the number of epoches')
    plt.ylabel('loss')
    os.makedirs(f'analysis',exist_ok=True)
    n=len(os.listdir('analysis'))
    plt.savefig(f'analysis/loss{n+1}.png')

    plt.figure()
    plt.title(f"lr:{args.learning_rate} momentum:{args.momentum} epoches:{args.epoches}")
    plt.plot(test_counter, acc_list, color='red')
    plt.legend(['Accuracy'], loc='upper right')
    plt.xlabel('the number of epoches')
    plt.ylabel('acc')
    plt.savefig(f'analysis/acc{n+1}.png')
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train an MLP")
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--logger", type=str, default="logs/logger.log", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--epoches", type=int, default=200, help="epoches")
    parser.add_argument("--num_cls", type=int, default=10, help="number of classes")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="learning rate")
    parser.add_argument("--mean", type=float, default=0.1307, help="average")
    parser.add_argument("--std", type=float, default=0.3081, help="variance")
    parser.add_argument("--dim_input", type=int, default=28*28, help="dim of input")
    parser.add_argument("--dim_hidden", type=int, default=1000, help="dim of hidden layer")

    args = parser.parse_args()
    os.makedirs('logs',exist_ok=True)
    train(args)