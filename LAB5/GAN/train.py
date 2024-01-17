from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from load_data import *
import argparse
from model import *
import os
from tqdm import trange,tqdm
import gif
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
def Gradient_Panelty(x_fake,x_fd):
    gradient = torch.autograd.grad(  
        outputs=x_fd,  
        inputs=x_fake,  
        grad_outputs=torch.ones_like(x_fd).to(x_fake.device),  
        create_graph=True,  
        retain_graph=True  
    )[0]
    gp = ((gradient.norm(2, dim=-1) - 1) ** 2).mean()
    return gp
def draw_loss(args,d_loss,g_loss):
    plt.figure()
    ax1=plt.subplot(211)
    ax1.plot(d_loss,label="discriminator loss")
    ax2=plt.subplot(212)
    ax2.plot(g_loss,label="generator loss")
    plt.savefig(f"results/{args.model}/loss-{args.model}-{args.optimizer}.png")
def train(args):
    os.makedirs(f"ckpt/{args.model}",exist_ok=True)
    os.makedirs(f"results/{args.model}",exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader=DataLoader(PointsDataset(data),batch_size=args.batch,shuffle=True)
    model_d=Discriminator() if args.model=="GAN" else WGAN_Discriminator()
    model_d.to(device)
    model_g=Generator(args.input_size)
    model_g.to(device)
    if args.optimizer=="adam":
        optim_D=optim.Adam(model_d.parameters(),lr=args.learning_rate,weight_decay=0.0005,betas=(0.5, 0.9))
        optim_G=optim.Adam(model_g.parameters(),lr=args.learning_rate,weight_decay=0.0005,betas=(0.5, 0.9))
    elif args.optimizer=="RMSProp":
        optim_D=optim.RMSprop(model_d.parameters(),lr=args.learning_rate,weight_decay=0.0005)
        optim_G=optim.RMSprop(model_g.parameters(),lr=args.learning_rate,weight_decay=0.0005)
    elif args.optimizer=="SGD":
        optim_D=optim.SGD(model_d.parameters(),lr=args.learning_rate,weight_decay=0.0005,momentum=0.9)
        optim_G=optim.SGD(model_g.parameters(),lr=args.learning_rate,weight_decay=0.0005,momentum=0.9)
    else:
        raise NotImplementedError("no such option")
    if args.model=="GAN":
        cri=nn.BCELoss()
    model_d.train()
    model_g.train()
    model_d.to(device)
    model_g.to(device)
    frame=[]
    d_loss=[]
    g_loss=[]
    for epoch in trange(args.epoches):
        model_d.train()
        model_g.train()
        d_loss_epoch=0.
        g_loss_epoch=0.
        counter=0
        for x in train_loader:
            b,_=x.shape
            #discriminator training
            z=torch.randn((b,args.input_size))
            #z=torch.rand((b,args.input_size))
            x_fake=model_g(z.to(device)).to(device)
            if args.model=="WGAN-GP":
                epsilon=torch.rand((b,2)).to(device)
                x_fake=epsilon*x.to(device)+(1-epsilon)*x_fake
            x_d=model_d(x.to(device)).to(device).view(-1)
            if args.model=="GAN":
                loss1=cri(x_d,torch.ones_like(x_d).to(device))
            elif args.model=="WGAN" or args.model=="WGAN-GP":
                loss1=-torch.mean(x_d)
            x_fd=model_d(x_fake.to(device)).to(device).view(-1)
            if args.model=="GAN":
                loss2=cri(x_fd,torch.zeros_like(x_fd).to(device))
            elif args.model=="WGAN" or args.model=="WGAN-GP":
                loss2=torch.mean(x_fd)
            if args.model=="GAN" or args.model=="WGAN":
                loss_d=loss1+loss2
            else:
                loss_gp=Gradient_Panelty(x_fake.to(device),x_fd)
                loss_d=loss1+loss2+loss_gp*args.lam
            optim_D.zero_grad()
            loss_d.backward(retain_graph=True)
            optim_D.step()
            for param in model_d.parameters():
                param.data.clamp_(-args.clamp,args.clamp)
            #generator training
            x_fd=model_d(x_fake.to(device)).to(device).view(-1)
            if args.model=="GAN":
                loss3=cri(x_fd,torch.ones_like(x_fd).to(device))
            elif args.model=="WGAN" or args.model=="WGAN-GP":
                loss3=-torch.mean(x_fd)
            optim_G.zero_grad()
            loss3.backward()
            optim_G.step()
            d_loss_epoch+=loss_d.item()
            g_loss_epoch+=loss3.item()
            counter+=1
        d_loss_epoch/=counter
        g_loss_epoch/=counter
        d_loss.append(d_loss_epoch)
        g_loss.append(g_loss_epoch)
        if (epoch+1)%20==0:
            f_tmp=evaluate(args,model_d,model_g,train_loader,epoch)
            frame.append(f_tmp)
    gif.save(frame,f"results/{args.model}/result-{args.model}-{args.optimizer}.gif",duration=10)
    draw_loss(args,d_loss,g_loss)
    statedict={'Discriminator':model_d.state_dict(),'Generator':model_g.state_dict()}
    torch.save(statedict,f"ckpt/{args.model}/best-{args.model}-{args.optimizer}.pth")
        
@gif.frame
def evaluate(args,model_d,model_g,train_loader,epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_g.eval()
    model_d.eval()
    real=[]
    z=torch.randn((10000,args.input_size))
    x_fake=model_g(z.to(device))
    fake=x_fake.tolist()
    #z=torch.rand((b,args.input_size))
    x_fake=model_g(z.to(device))
    for x in train_loader:
        b,_=x.shape
        # print(x_fake)
        real.extend(x.tolist())
    real_x=[i[0] for i in real]
    real_y=[i[1] for i in real]
    fake_x=[i[0] for i in fake]
    fake_y=[i[1] for i in fake]
    l=min(real_x)-0.5
    r=max(real_x)+0.5
    b=min(real_y)-0.5
    t=max(real_y)+0.5
    plt.figure()
    bg=[]
    cood=[]
    x=l
    y=b
    X=[]
    Y=[]
    while x<r:
        X.append(x)
        x+=0.01
    while y<t:
        Y.append(y)
        y+=0.01
    cood=list(product(X,Y))
    cood_tmp =[list(i) for i in cood]
    cood_tmp=torch.FloatTensor(cood_tmp).to(device)
    conf=model_d(cood_tmp).view(-1)
    conf=conf.tolist()
    # print(len(cood))
    # print(len(X))
    X,Y=zip(*cood)
    plt.scatter(X,Y,c=conf,cmap="Greys")
    plt.colorbar()
    plt.scatter(real_x,real_y,c="red",label="real data")
    plt.scatter(fake_x,fake_y,c="blue",label="fake data")
    plt.xlim((l,r))
    plt.ylim((b,t))
    plt.title(f"epoch{epoch+1}")
    # plt.show()
    


if __name__=="__main__":
    parser = argparse.ArgumentParser("lab4")
    parser.add_argument("--model", default="GAN")  # GAN WGAN WGAN-GP
    parser.add_argument("--epoches", type=int,default=30)
    parser.add_argument("--batch", type=int,default=128)  
    parser.add_argument("--input_size", default=64, type=int) 
    parser.add_argument("--learning_rate",type=float,default=5e-4,help="you know")
    parser.add_argument("--clamp", type=float,default=1)
    parser.add_argument("--lam", type=float,default=0.001)
    parser.add_argument("--optimizer", type=str,default="RMSProp")
    args = parser.parse_args()
    train(args)