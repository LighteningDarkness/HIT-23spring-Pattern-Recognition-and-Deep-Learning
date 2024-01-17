from torch import nn

class MLP(nn.Module):     
    def __init__(self,args):
        super(MLP,self).__init__()
        self.args=args
        self.linear1=nn.Linear(args.dim_input,args.dim_hidden)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(args.dim_hidden,args.dim_hidden) #2个隐层
        self.relu2=nn.ReLU()
        self.linear3=nn.Linear(args.dim_hidden,args.num_cls)
  
    def forward(self, x):
        x=x.view(-1,self.args.dim_input)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
