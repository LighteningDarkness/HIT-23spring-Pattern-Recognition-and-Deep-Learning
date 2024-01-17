import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

class PlantSeedlingsDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.labels = []
        self.image_paths = []
        if self.train:
            for label in os.listdir(root_dir + "/train/"):
                if label not in [".DS_Store", "desktop.ini"]:
                    for file_name in os.listdir(root_dir + "/train/" + label):
                        if file_name.endswith('.png'):
                            self.labels.append(label)
                            self.image_paths.append(os.path.join(self.root_dir, "train", label, file_name))
        else:
            for file_name in os.listdir(root_dir + "/test/"):
                if file_name.endswith('.png'):
                    self.image_paths.append(os.path.join(self.root_dir, "test", file_name))

        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.train:
            image = Image.open(self.image_paths[index]).convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

            label = self.labels[index]
            label_idx = self.label_to_idx[label]

            return image, label_idx
        else:
            image = Image.open(self.image_paths[index]).convert("RGB")
            name = self.image_paths[index].split('/')[-1]
            if self.transform is not None:
                image = self.transform(image)

            return image, name


if __name__ == "__main__":
    data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),    
])
    dataset = PlantSeedlingsDataset(root_dir='.\plant-seedlings-classification', transform=data_transforms)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for images, labels in dataloader:
        print(images.shape, labels.shape)
        break