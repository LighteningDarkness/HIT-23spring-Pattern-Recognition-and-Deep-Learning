from torch.utils.data import DataLoader
from load_data import *
from torch import nn, optim
from model import *
import os
from tqdm import trange,tqdm
from sklearn.metrics import classification_report
import argparse
import matplotlib.pyplot as plt
from utils import *
from climate_contrast import *
def visualize(train_losses,test_losses,precision,recall,fscore,name):
    plt.figure()
    plt.plot(train_losses, label='train_loss')
    plt.plot(test_losses, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(name+'loss.png')

    max_p=max(precision)
    max_p_idx=precision.index(max_p)
    plt.figure()
    plt.plot(precision)
    plt.scatter(max_p_idx, max_p, color='red', s=50)  # 在最大值点上绘制一个红色的圆点
    plt.annotate(f'max: ({max_p_idx:.2f}, {max_p:.2f})', xy=(max_p_idx, max_p), xytext=(max_p_idx, max_p))
    plt.xlabel('epoch')
    plt.ylabel('precision')
    plt.legend()
    plt.savefig(name+'precision.png')

    max_r=max(recall)
    max_r_idx=recall.index(max_r)
    plt.figure()
    plt.plot(recall)
    plt.scatter(max_r_idx, max_r, color='red', s=50)  # 在最大值点上绘制一个红色的圆点
    plt.annotate(f'max: ({max_r_idx:.2f}, {max_r:.2f})', xy=(max_r_idx, max_r), xytext=(max_r_idx, max_r))
    plt.xlabel('epoch')
    plt.ylabel('recall')
    plt.legend()
    plt.savefig(name+'recall.png')

    max_f=max(fscore)
    max_f_idx=fscore.index(max_f)
    plt.figure()
    plt.plot(fscore)
    plt.scatter(max_f_idx, max_f, color='red', s=50)  # 在最大值点上绘制一个红色的圆点
    plt.annotate(f'max: ({max_f_idx:.2f}, {max_f:.2f})', xy=(max_f_idx, max_f), xytext=(max_f_idx, max_f))
    plt.xlabel('epoch')
    plt.ylabel('f1-score')
    plt.legend()
    plt.savefig(name+'fscore.png')
def train_shopping(args):

    train_dataset,test_dataset,vocab=build_shopping()
    config.vocab=vocab
    train_loader=DataLoader(train_dataset,batch_size=args.batch,shuffle=True,collate_fn=ShoppingDataset.collect_fn)
    test_loader=DataLoader(test_dataset,batch_size=args.batch,shuffle=True,collate_fn=ShoppingDataset.collect_fn)

    net=TextClassification(args,vocab)
    optimizer=optim.Adam(net.parameters(),lr=args.learning_rate,weight_decay=0.0005)
    net.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    net.to(device)
    os.makedirs(f'ckpt/shopping/{args.model}',exist_ok=True)
    os.makedirs(f'results/shopping/{args.model}',exist_ok=True)
    cri=nn.CrossEntropyLoss()
    max_f=0.
    train_losses=[]
    test_losses=[]
    precision=[]
    recall=[]
    fscore=[]
    for epoch in trange(args.epoches,desc='training'):
        loss_epoch=0
        for (x,y) in tqdm(train_loader,leave=False,desc='data loading'):
            #print(data.shape)
            output=net(x).to(device)
            optimizer.zero_grad()
            target_onehot=nn.functional.one_hot(y,args.num_cls).float().to(device)
            loss=cri(output,target_onehot)
            loss.backward()
            optimizer.step()
            loss_epoch+=loss.item()
        loss_epoch/=len(train_loader.dataset)
        result,val_loss=evaluate_shopping(net,test_loader)
        train_losses.append(loss_epoch)
        test_losses.append(val_loss)
        precision.append(result['macro avg']['precision'])
        recall.append(result['macro avg']['recall'])
        fscore.append(result['macro avg']['f1-score'])
        if result['macro avg']['f1-score']>max_f:
            max_f=result['macro avg']['f1-score']
            torch.save(net.state_dict(),f'ckpt/shopping/{args.model}/best.pth')
    visualize(train_losses,test_losses,precision,recall,fscore,f'results/shopping/{args.model}/')
def evaluate_shopping(net,val_loader):

    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    test_loss = 0
    pred_list=[]
    gt_list=[]
    sum=0
    cri=nn.CrossEntropyLoss()
    with torch.no_grad():
        for (x,y) in tqdm(val_loader,desc="testing",leave=False):
            output = net(x.to(device)).to(device)            
            target_onehot=nn.functional.one_hot(y.to(device),args.num_cls).float().to(device)
            test_loss += cri(output,target_onehot).item()
            pred=output.argmax(dim=-1)
            pred_list.extend(pred.tolist())
            gt_list.extend(y.tolist())
    test_loss /= len(val_loader.dataset)
    result=classification_report(gt_list, pred_list, target_names=shopping_cats,output_dict=True)
    return result,test_loss


def train_climate(args):

    train_loader,test_loader=getWeatherData(args)
    net=JenaNet(args)

    optimizer=optim.Adam(net.parameters(),lr=args.learning_rate,weight_decay=0.0005,betas=(0.9,0.999))

    net.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    net.to(device)
    os.makedirs(f'ckpt/climate',exist_ok=True)
    os.makedirs(f'results/climate/contrast',exist_ok=True)
    #cri=nn.MSELoss(reduction='sum')
    cri=nn.L1Loss(reduction='sum')
    if args.test:
        
        error=contrast(net,test_loader,args)
        avg=sum(error,0)/len(error)
        error=sorted(error)
        med=error[len(error)//2]
        print(f"avg:{avg}  median:{med}")
        return
    train_losses=[]
    test_losses=[]
    min_loss=-1.
    for epoch in trange(args.epoches):
        loss_epoch=0
        for (x,y) in tqdm(train_loader,leave=False,desc="data loading"):
            #print(data.shape)
            h0=torch.rand((x.shape[0],args.hidden_size)).to(device)
            c0=torch.rand((x.shape[0],args.hidden_size)).to(device)
            x=x.transpose(1,0).to(device)
            output=net(x,h0,c0).to(device)
            optimizer.zero_grad()
            loss=cri(output,y.to(device))
            loss.backward()
            optimizer.step()
            # print(net.linear.weight)
            loss_epoch+=loss.item()/288
        loss_epoch/=(len(train_loader.dataset))
        val_loss,out_eval,yt=evaluate_climate(net,test_loader,args)
        train_losses.append(loss_epoch)
        test_losses.append(val_loss)
        print(f"train loss:{loss_epoch} val loss:{val_loss}")
        if min_loss<0 or val_loss<min_loss:
            min_loss=val_loss
            torch.save(net.state_dict(),f'ckpt/climate/best.pth')
            out_best=out_eval
    plt.figure()
    plt.plot(train_losses, label='train_loss')
    plt.plot(test_losses, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('results/climate/loss.png')

    error0=out_best.clone().cpu().detach()-yt
    error0=torch.flatten(error0).tolist()
    avg=sum(error0,0)/len(error0)
    error=sorted(error0)
    med=error[len(error0)//2]
    print(f"avg:{avg}  median:{med}")

    for i in range(out_best.shape[0]):
        tmp=out_best[i].cpu().detach()
        yi=yt[i]
        plt.figure()
        plt.plot(tmp.tolist(), label='predicted value')
        plt.plot(yi.tolist(), label='real value')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(f'results/climate/contrast/contrast{i}.png')
        plt.close()


def evaluate_climate(net,val_loader,args):

    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    test_loss = 0
    pred_list=[]
    gt_list=[]
    sum=0
    #cri=nn.MSELoss(reduction='sum')
    cri=nn.L1Loss(reduction='sum')
    with torch.no_grad():
        for (x,y) in tqdm(val_loader,desc="testing",leave=False):
            h0=torch.zeros((x.shape[0],args.hidden_size)).to(device)
            c0=torch.zeros((x.shape[0],args.hidden_size)).to(device)
            x=x.transpose(1,0).to(device)
            output = net(x,h0,c0).to(device)
            test_loss += cri(output,y.to(device)).item()/288
    test_loss /= (len(val_loader.dataset))
    return test_loss,output,y
def train(args):
    if args.task=='shopping':
        train_shopping(args)
    else:
        train_climate(args)
if __name__=="__main__":
    parser = argparse.ArgumentParser("lab4")
    parser.add_argument("--model", default="GRU")  # RNN GRU LSTM Bi-LSTM
    parser.add_argument("--epoches", type=int,default=30)
    parser.add_argument("--batch", type=int,default=128)  
    parser.add_argument("--task", type=str,default="climate")  # shopping or climate
    parser.add_argument("--output_size", type=int,default=288)
    parser.add_argument("--hidden_size", type=int,default=512)
    parser.add_argument("--input_size", default=21, type=int) 
    parser.add_argument("--learning_rate",type=float,default=1e-3,help="you know")
    parser.add_argument("--num_cls", default=10, type=int)
    parser.add_argument("--test",action='store_true')
    args = parser.parse_args()
    train(args)