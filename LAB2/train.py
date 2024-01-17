from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from AlexNet import *
import os
import argparse
from load_data import *
from tqdm import trange,tqdm
from torch.utils.tensorboard import SummaryWriter  
def onehot(target,num_cls):
    l=[]
    for y in target:
        tmp=torch.zeros(num_cls)
        tmp[y.item()]=1
        tmp=tmp.unsqueeze(0)
        l.append(tmp)
    return torch.cat(l,dim=0)

def train(args):
 
    os.makedirs('data',exist_ok=True)

    if args.train:
        data_train = MyDataset(split='train')
        data_val = MyDataset(split='val')
        train_loader = DataLoader(dataset=data_train,
                                batch_size = args.batch,
                                shuffle = True)
        val_loader = DataLoader(dataset=data_val,
                            batch_size = args.batch,
                            shuffle = True)
        net=AlexNet()
        if args.opt=="sgd":
            optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum,weight_decay=0.0005)
        elif args.opt=="adam":
            optimizer = optim.Adam(net.parameters(),lr=args.learning_rate,weight_decay=0.0005)
        else:
            raise NotImplementedError("no such optimizer")
        net.train()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print(device)
        net.to(device)
        os.makedirs('ckpt',exist_ok=True)
        cri=nn.CrossEntropyLoss()
        writer = SummaryWriter(args.logger)
        max_acc=0.
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
            loss_epoch/=len(train_loader.dataset)
            acc,val_loss=evaluate(net,val_loader)
            writer.add_scalars("loss",{"training loss":loss_epoch,"val loss":val_loss},epoch)
            writer.add_scalar("accuracy",acc,epoch)
            if (epoch+1)%10==0:
                torch.save(net.state_dict(),f'ckpt/epoch{epoch+1}.pth')
                if acc>max_acc:
                    max_acc=acc
                    torch.save(net.state_dict(),f'ckpt/best.pth')
    if args.test:
        data_test = MyDataset(split='test')
        test_loader = DataLoader(dataset=data_test,
                            batch_size = args.batch,
                            shuffle = True)
        net=AlexNet()
        net.load_state_dict(torch.load(args.ckpt))
        # net=torch.load('ckpt/epoch90sgd.pth')
        print(type(net))
        acc,loss=evaluate(net,test_loader)
        print(f"acc:{acc},loss:{loss}")

def evaluate(net,val_loader):

    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    test_loss = 0

    sum=0
    cri=nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in tqdm(val_loader,desc="testing",leave=False):
            output = net(data.to(device))
            target_onehot=onehot(target,args.num_cls)
            test_loss += cri(output,target_onehot.to(device)).item()
            sum+=(output.argmax(dim=1)==target.to(device)).sum().cpu().item()
    test_loss /= len(val_loader.dataset)
    acc=sum/len(val_loader.dataset)
    return acc,test_loss

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train an AlexNet for classification")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--epoches", type=int, default=30, help="epoches")
    parser.add_argument("--num_cls", type=int, default=101, help="number of classes")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="learning rate")
    parser.add_argument("--logger", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--train",action='store_true',  help="train?")
    parser.add_argument("--test",action='store_true',  help="test?")
    parser.add_argument('--opt',choices=["sgd","adam"],type=str,help='optimizer')
    parser.add_argument("--ckpt", type=str, default="ckpt/best.pth", help="ckpt for testing")
    os.makedirs('logs',exist_ok=True)
    args = parser.parse_args()
    train(args)