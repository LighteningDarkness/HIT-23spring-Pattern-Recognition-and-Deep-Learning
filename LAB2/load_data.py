from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
caltech_path='data/caltech-101/101_ObjectCategories'
classes=os.listdir(caltech_path)
classes.remove('BACKGROUND_Google')

class MyDataset(Dataset):
    def __init__(self,path=caltech_path,split='train'):
        super(MyDataset,self).__init__()
        self.transform=transforms.Compose(
            [
                transforms.Resize((227,227)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ]
        )
        self.split=split
        self.images=[]
        self.labels=[]
        for i,c in enumerate(classes):
            path_tmp=os.path.join(path,c)
            img_list=os.listdir(path_tmp)
            for j,img in enumerate(img_list):
                if (split=='train' and j<int(len(img_list)*0.8)) or (split=='val' and j>=int(len(img_list)*0.8) and j<int(len(img_list)*0.9)) or (split=='test' and j>=int(len(img_list)*0.9)):
                    image=Image.open(os.path.join(path_tmp,img)).convert('RGB')
                    image=self.transform(image)
                    self.images.append(image)
                    self.labels.append(i)

                
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return self.images[index],self.labels[index]
    
if __name__=="__main__":
    from AlexNet import AlexNet
    model=AlexNet()
    md=MyDataset(split='val')
    print(model(md[0][0]).shape)



