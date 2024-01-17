from torch import nn
import torch

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet,self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4), 
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2), 
            nn.MaxPool2d(3, 2),  
            nn.Conv2d(96, 256, 5, padding=2),  
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(3, 2),  
            nn.Conv2d(256, 384, 3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  
        )
        self.classifier = nn.Sequential(
            
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=101),
        )
    def forward(self,x):
        x = self.feature_extractor(x)
        x = x.view(-1, 256 * 6 * 6)  
        return self.classifier(x)
    

