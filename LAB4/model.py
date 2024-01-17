import torch
from torch import nn
import math
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(RNN,self).__init__()
        self.ih=nn.Linear(input_size,hidden_size)
        self.hh=nn.Linear(hidden_size,hidden_size)

        self.tanh=nn.Tanh()
    def forward(self,x,h0):
        #(L,H_in)
        if len(x.shape)==2:
            h_list=[h0]
            for j in range(x.shape[0]):
                xi=x[j:j+1,:]
                h=h_list[-1]
                ht=self.tanh(self.ih(xi)+self.hh(h))
                h_list.append(ht)
            h_list.pop(0)
            return torch.cat(h_list,dim=0)
        #(L,N,H_in)
        elif len(x.shape)==3:
            h_list=[h0]
            for xi in x:
                h=h_list[-1]
                ht=self.tanh(self.ih(xi)+self.hh(h))
                h_list.append(ht)
            h_list.pop(0)
            return torch.stack(h_list)
        else:
            raise NotImplementedError('input must be 2 or 3 dim')

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,bidirection=False):
        super(LSTM,self).__init__()
        self.bidirection=bidirection
        self.ii=nn.Linear(input_size,hidden_size)   
        self.hi=nn.Linear(hidden_size,hidden_size)
        self.i2f=nn.Linear(input_size,hidden_size)
        self.hf=nn.Linear(hidden_size,hidden_size)
        self.ig=nn.Linear(input_size,hidden_size)
        self.hg=nn.Linear(hidden_size,hidden_size)
        self.io=nn.Linear(input_size,hidden_size)
        self.ho=nn.Linear(hidden_size,hidden_size)
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
    def forward(self,x,h0,c0):
        if not self.bidirection:
            #(L,H_in)
            if len(x.shape)==2:
                h_list=self.forward_single(x,h0,c0)
                return torch.cat(h_list,dim=0)
            elif len(x.shape)==3:
                #(L,N,H_in)
                h_list=[h0]
                c_list=[c0]
                for xi in x:
                    h=h_list[-1]
                    c=c_list[-1]
                    it=self.sigmoid(self.ii(xi)+self.hi(h))
                    ft=self.sigmoid(self.i2f(xi)+self.hf(h))
                    gt=self.tanh(self.ig(xi)+self.hg(h))
                    ot=self.sigmoid(self.io(xi)+self.ho(h))
                    ct=ft*c+it*gt
                    ht=ot*self.tanh(ct)
                    h_list.append(ht)
                    c_list.append(ct)
                h_list.pop(0)
                return torch.stack(h_list)
        else:
            if len(x.shape)==2:
                output_bi=[]
                h_list=self.forward_single(x,h0,c0)
                x_flip=torch.fliplr(x)
                h_list_backward=self.forward_single(x_flip,h0,c0)
                assert len(h_list)==len(h_list_backward)
                for i in range(len(h_list)):
                    output_bi.append(torch.cat((h_list[i],h_list_backward[i]),dim=1))
                return torch.cat(output_bi,dim=0)
            elif len(x.shape)==3:
                #(L,N,H_in)
                h_list=[h0]
                c_list=[c0]
                output_bi=[]
                for xi in x:
                    h=h_list[-1]
                    h=h_list[-1]
                    c=c_list[-1]
                    it=self.sigmoid(self.ii(xi)+self.hi(h))
                    ft=self.sigmoid(self.i2f(xi)+self.hf(h))
                    gt=self.tanh(self.ig(xi)+self.hg(h))
                    ot=self.sigmoid(self.io(xi)+self.ho(h))
                    ct=ft*c+it*gt
                    ht=ot*self.tanh(ct)
                    h_list.append(ht)
                    c_list.append(ct)
                x_flip=torch.fliplr(x)
                h_list_flip=[h0]
                c_list_flip=[c0]
                for xi in x:
                    h=h_list_flip[-1]
                    c=c_list_flip[-1]
                    it=self.sigmoid(self.ii(xi)+self.hi(h))
                    ft=self.sigmoid(self.i2f(xi)+self.hf(h))
                    gt=self.tanh(self.ig(xi)+self.hg(h))
                    ot=self.sigmoid(self.io(xi)+self.ho(h))
                    ct=ft*c+it*gt
                    ht=ot*self.tanh(ct)
                    h_list_flip.append(ht)
                    c_list_flip.append(ct)
                h_list.pop(0)
                h_list_flip.pop(0)
                assert len(h_list)==len(h_list_flip)
                for i in range(len(h_list)):
                    output_bi.append(torch.cat((h_list[i],h_list_flip[len(h_list)-1-i]),dim=1))
                return torch.stack(output_bi)
            else:
                raise NotImplementedError('must be 2 or 3 dim')
    def forward_single(self,x,h0,c0):
        #(L,H_in)
        h_list=[h0]
        c_list=[c0]
        for j in range(x.shape[0]):
            xi=x[j:j+1,:]
            h=h_list[-1]
            c=c_list[-1]
            it=self.sigmoid(self.ii(xi)+self.hi(h))
            ft=self.sigmoid(self.i2f(xi)+self.hf(h))
            gt=self.tanh(self.ig(xi)+self.hg(h))
            ot=self.sigmoid(self.io(xi)+self.ho(h))
            ct=ft*c+it*gt
            ht=ot*self.tanh(ct)
            h_list.append(ht)
        h_list.pop(0)
        return h_list
class GRU(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(GRU,self).__init__()
        self.ir=nn.Linear(input_size,hidden_size)
        self.hr=nn.Linear(hidden_size,hidden_size)
        self.iz=nn.Linear(input_size,hidden_size)
        self.hz=nn.Linear(hidden_size,hidden_size)
        self.i2n=nn.Linear(input_size,hidden_size)
        self.hn=nn.Linear(hidden_size,hidden_size)
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
    def forward(self,x,h0):
        if len(x.shape)==2:
            return self.forward_single(x,h0)
        elif len(x.shape)==3:
            #(L,N,H_in)
            h_list=[h0]
            for xi in x:
                h=h_list[-1]
                rt=self.sigmoid(self.ir(xi)+self.hr(h))
                zt=self.sigmoid(self.iz(xi)+self.hz(h))
                nt=self.tanh(self.i2n(xi)+rt*self.hn(h))
                ht=(1-zt)*nt+zt*h
                h_list.append(ht)
            h_list.pop(0)
            return torch.stack(h_list)
        else:
            raise NotImplementedError('must be 2 or 3 dim')
    def forward_single(self,x,h0):
        #(L,H_in)
        h_list=[h0]
        for j in range(x.shape[0]):
            xi=x[j:j+1,:]
            h=h_list[-1]
            rt=self.sigmoid(self.ir(xi)+self.hr(h))
            zt=self.sigmoid(self.iz(xi)+self.hz(h))
            nt=self.tanh(self.i2n(xi)+rt*self.hn(h))
            ht=(1-zt)*nt+zt*h
            h_list.append(ht)
        h_list.pop(0)
        output_list=[self.ho(h) for h in h_list]
        return torch.cat(output_list,dim=0)
class TextClassification(nn.Module):
    def __init__(self,args,vocab):
        super().__init__()
        self.args=args
        if args.model=='RNN':
            self.model=RNN(args.input_size,args.hidden_size)
        elif args.model=='LSTM':
            self.model=LSTM(args.input_size,args.hidden_size)
        elif args.model=='BiLSTM':
            self.model=LSTM(args.input_size,args.hidden_size,bidirection=True)
        else:
            self.model=GRU(args.input_size,args.hidden_size)
        self.linear=nn.Linear(args.hidden_size,args.output_size) if args.model!='BiLSTM' else nn.Linear(2*args.hidden_size,args.output_size)
        self.embedding=nn.EmbeddingBag(num_embeddings=len(vocab),embedding_dim=args.input_size)
    def forward(self,x):
        L,N=x.shape
        x=x.reshape(L*N,-1)
        x=self.embedding(x)
        x=x.reshape(L,N,-1)
        h0=torch.zeros((N,self.args.hidden_size)).to(x.device)
        c0=torch.zeros((N,self.args.hidden_size)).to(x.device)
        if self.args.model=='LSTM' or self.args.model=='BiLSTM':
            output=self.model(x,h0,c0)
        else:
            output=self.model(x,h0)
        output=torch.mean(output,dim=0) if self.args.model=="RNN" else output[-1]
        #output=output[-1]
        return self.linear(output)
class ClimateNet(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.model=LSTM(21,args.hidden_size)
        self.linear=nn.Linear(args.hidden_size,1048)
        self.linear2=nn.Linear(1048,288)
        self.relu=nn.ReLU()
        self.reset_param()
    def forward(self,x,h0,c0):
        h=self.model(x,h0,c0)
        h=h[-1]*10
        h=self.linear(h)
        h=self.relu(h)
        return self.linear2(h)
    def reset_param(self):
        std = 1.0 / math.sqrt(self.args.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


if __name__=="__main__":
    x=torch.rand((3,3,100))
    length=[1,2,3]
    h0=torch.zeros((1,200))
    c0=torch.zeros((1,200))
    # rnn=RNN(100,200,10)
    # print(rnn(x,h0)[0].shape)
    # lstm=LSTM(100,200,10,bidirection=True)
    # print(lstm(x,h0,c0)[0].shape)
    gru=GRU(100,200,10)
    print(gru.modules)