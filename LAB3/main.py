from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from model import ResNet18withSE, ResNet18, VGG11
from dataset import PlantSeedlingsDataset
import torch
import yaml
import pandas as pd

from main_worker import train, test
from draw import visualize

#使用命令行参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yml', help='config directory')
parser.add_argument('--train', type=str, default=True, help='train or test')

args = parser.parse_args()
# 定义数据预处理transform
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if args.train:
    #读取config.yml
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    #设置设备为config中相应的设备，如果没有则使用cpu，用于后续的to(device)
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    epoch = config['epoch']
    save_path = config['save_path']+args.config.split('\\')[-1].split('.')[0]+'.pth'
    batch_size = config['batch_size']
    model = config['model']
    optimizer = config['optimizer']
    lr = config['lr']
    momentum = config['momentum']
    tvrate = config['rate']
    result_path = config['result_path']
    data_dir = config['data_dir']


    # 创建数据集对象，并设置训练集和验证集的比例
    dataset = PlantSeedlingsDataset(data_dir, transform=data_transform)
    train_len = int(len(dataset) * tvrate)
    train_dataset, val_dataset = random_split(dataset, [train_len, len(dataset) - train_len])

    # 创建DataLoader对象
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型对象
    if model == 'ResNet18withSE':
        model = ResNet18withSE(12)
    elif model == 'ResNet18':
        model = ResNet18(12)
    elif model == 'VGG11':
        model = VGG11(12)

    # 将模型移动到device上
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # 训练模型
    accuracy_list, loss_list, val_loss_list = train(train_dataloader, 
                                                    val_dataloader, 
                                                    optimizer, 
                                                    criterion, 
                                                    model, 
                                                    epoch=epoch,
                                                    save=save_path)
    #画图
    visualize(loss_list, val_loss_list, accuracy_list, result_path, args.config.split('.')[0] )
else:
    #读取config.yml
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    #设置设备为config中相应的设备，如果没有则使用cpu，用于后续的to(device)
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    batch_size = config['batch_size']
    model = config['model']
    data_dir = config['data_dir']
    model_path = config['save_path']+args.config.split('.')[0]+'.pth'
    result_path = config['result_path']

    # 创建模型对象
    if model == 'ResNet18withSE':
        model = ResNet18withSE(12)
    elif model == 'ResNet18':
        model = ResNet18(12)
    elif model == 'VGG11':
        model = VGG11(12)
    # 将模型移动到device上
    model = model.to(device)
    # 加载模型参数
    model.load_state_dict(torch.load(model_path))

    test_dataset = PlantSeedlingsDataset(data_dir, transform=data_transform, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    result = test(test_dataloader, model)

    # 保存结果至csv文件中，file和species两列
    
    df = pd.DataFrame(result, columns=['file', 'species'])
    df.to_csv('submission.csv', index=False)
