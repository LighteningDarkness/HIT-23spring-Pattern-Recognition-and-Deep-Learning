from torch.utils.data import DataLoader
from tqdm import tqdm
from load_data import *
from torch import nn
import logging
import argparse
from BERT import *
import os
from transformers import BertTokenizer
from transformers import BertConfig, BertForSequenceClassification
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def evaluate(model,test_loader,length):
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model=model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_name)
    sum=0
    for i,(x,y) in enumerate(test_loader):
        x=tokenizer(x, padding=True, truncation=True, return_tensors='pt').to(device)
        y=y.to(device)
        output=model(**x,labels=y)
        output=output[1]
        sum+=(output.argmax(dim=1) == y.argmax(dim=1)).sum()
    return sum/length

def train(args):
    config = BertConfig.from_pretrained(args.pretrained_name, num_labels=args.num_cls, hidden_dropout_prob=args.hidden_dropout_prob)
    model = BertForSequenceClassification.from_pretrained(args.pretrained_name, config=config)
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model.to(device)
    model.train()
    max_acc=0.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_name)
    data,lb=readcsv(args.data,100000)
    train_dataset=CommentsDataset(data,lb,split='train')
    test_dataset=CommentsDataset(data,lb,split='test')
    train_loader=DataLoader(train_dataset,batch_size=args.batch)
    test_loader=DataLoader(test_dataset,batch_size=args.batch)
    logger = get_logger(args.logger)
    loss_func=loss=nn.CrossEntropyLoss()
    os.makedirs('model/BERT',exist_ok=True)
    logger.info('start training!')
    for epoch in tqdm(range(args.epoches)):
        for i,(x,y) in tqdm(enumerate(train_loader),desc="batch",leave=False):
            x=tokenizer(x, padding=True, truncation=True, return_tensors='pt').to(device)
            y=y.to(device)

            output=model(**x,labels=y)
            output=output[1]
            optimizer.zero_grad()
            
            loss=loss_func(output,y)

            loss.backward()
            optimizer.step()
        acc=evaluate(model,test_loader,len(test_dataset))
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.epoches, loss, acc ))

        torch.save(model.state_dict(),f'model/BERT/epoch{epoch}.pth')

        if acc>max_acc:
            max_acc=acc
            torch.save(model.state_dict(),f'model/BERT/best.pth')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Finetuning the BERT model for sentimental analysis.")
    parser.add_argument("--pretrained_name",type=str,default='bert-base-uncased', help="name of BERT pretrained model")
    parser.add_argument("--batch", type=int, default=12, help="batch size")
    parser.add_argument("--logger", type=str, default="logs/logger.log", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--epoches", type=int, default=10, help="epoches")
    parser.add_argument("--num_cls", type=int, default=5, help="number of classes")
    parser.add_argument("--data", default="data/Amazon_Unlocked_Mobile.csv",type=str,help="Path to data")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="as the name")

    args = parser.parse_args()
    os.makedirs('logs',exist_ok=True)
    train(args)
