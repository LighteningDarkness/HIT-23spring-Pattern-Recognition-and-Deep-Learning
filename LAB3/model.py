import torch.nn as nn
import torch.nn.functional as F
import torch

class VGG11(nn.Module):
    def __init__(self,num_cls):
        super(VGG11,self).__init__()
        self.num_cls=num_cls
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
             nn.Linear(512*7*7, 4096),
             nn.ReLU(),
             nn.Dropout(0.5),
             nn.Linear(4096, 4096),
             nn.ReLU(),
             nn.Dropout(0.5),
             nn.Linear(4096, self.num_cls)
         )
    
    def forward(self,x):
        x=self.feature_extractor(x)
        x=x.view(-1,512*7*7)
        x=self.classifier(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride,max_pool=False):
        super(ResNetBlock,self).__init__()
        self.max_pool=nn.MaxPool2d(3,2,padding=1) if max_pool else None
        self.seq=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,stride=stride,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel,3,stride=1,padding=1),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut=nn.Conv2d(in_channel,out_channel,1,stride=2) if stride!=1 or max_pool else None

    def forward(self,x):
        
        out=x if self.max_pool is None else self.max_pool(x)
        x= x if self.shortcut is None else self.shortcut(x)
        out=self.seq(out)
        out=out+x
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self,num_cls):
        super(ResNet18,self).__init__()
        self.conv1=nn.Conv2d(3,64,7,2,padding=3)
        self.conv2=ResNetBlock(64,64,stride=1,max_pool=True)
        self.conv3=ResNetBlock(64,128,stride=2)
        self.conv4=ResNetBlock(128,256,stride=2)
        self.conv5=ResNetBlock(256,512,stride=2)
        self.conv=nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )
        self.avg=nn.AvgPool2d(7)
        self.linear=nn.Linear(512,num_cls)
    
    def forward(self,x):
        x=self.conv(x)
        x=self.avg(x)
        x=x.view(-1,512)
        x=self.linear(x)
        return x
class SEBlock(nn.Module):
    def __init__(self,out_channel,r=16):
        super(SEBlock,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(out_channel, out_channel//r),   
            nn.ReLU(),
            nn.Linear(out_channel//r, out_channel),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_=x.shape
        se_out=self.avg_pool(x).view(b,c)
        se_out=self.linear(se_out)
        return se_out
class ResNetBlockwithSE(ResNetBlock):
    def __init__(self,in_channel,out_channel,stride,max_pool=False,r=16):
        super().__init__(in_channel,out_channel,stride,max_pool)
        self.se=SEBlock(out_channel,r=r)

    def forward(self, x):
        out=x if self.max_pool is None else self.max_pool(x)
        x= x if self.shortcut is None else self.shortcut(x)
        out=self.seq(out)
        se_out=self.se(out)
        se_out=se_out.view(se_out.shape[0],se_out.shape[1],1,1)
        out=out*se_out
        out=out+x
        return F.relu(out) 

class ResNet18withSE(nn.Module):
    def __init__(self,num_cls):
        super(ResNet18withSE,self).__init__()
        self.conv1=nn.Conv2d(3,64,7,2,padding=3)
        self.se=SEBlock(64)
        self.conv2=ResNetBlockwithSE(64,64,stride=1,max_pool=True)
        self.conv3=ResNetBlockwithSE(64,128,stride=2)
        self.conv4=ResNetBlockwithSE(128,256,stride=2)
        self.conv5=ResNetBlockwithSE(256,512,stride=2)
        self.conv=nn.Sequential(
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )
        self.avg=nn.AvgPool2d(7)
        self.linear=nn.Linear(512,num_cls)
    
    def forward(self,x):
        x=self.conv1(x)
        se_x=self.se(x)
        se_x=se_x.view(se_x.shape[0],se_x.shape[1],1,1)
        x=x*se_x
        x=self.conv(x)
        x=self.avg(x)
        x=x.view(-1,512)
        x=self.linear(x)
        return x

if __name__=="__main__":
    x=torch.rand((1,3,224,224))
    model=ResNet18withSE(12)
    x=model(x)
    print(x)



