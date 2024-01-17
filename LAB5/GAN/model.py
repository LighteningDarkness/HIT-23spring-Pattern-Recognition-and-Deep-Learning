from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.generator=nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(True),
            nn.Linear(128,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            nn.Linear(512,2)
        )

    def forward(self,x):
        return self.generator(x)
    

    
class WGAN_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator=nn.Sequential(
                nn.Linear(2,256),
                nn.LeakyReLU(),
                nn.Linear(256,256),
                nn.LeakyReLU(),
                nn.Linear(256,256),
                nn.LeakyReLU(),
                nn.Linear(256,1),
            )
    def forward(self,x):
        return self.discriminator(x)
    
class Discriminator(WGAN_Discriminator):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return nn.Sigmoid()(self.discriminator(x))