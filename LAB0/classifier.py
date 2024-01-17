from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizer
import torch
import argparse
def predict(string,args):
    config = BertConfig.from_pretrained(args.pretrained_name, num_labels=args.num_cls, hidden_dropout_prob=args.hidden_dropout_prob)
    model = BertForSequenceClassification.from_pretrained(args.pretrained_name, config=config)
    model.load_state_dict(torch.load(args.ckpt))
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_name)

    x=tokenizer([string], padding=True, truncation=True, return_tensors='pt').to(device)
    output=model(**x)

    y=output[0].argmax(dim=1)
    print(f"Rating score is {y[0].item()+1}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Use the fine-tuning checkpoint to predict rating.")
    parser.add_argument("--pretrained_name",type=str,default='bert-base-uncased', help="name of BERT pretrained model")
    parser.add_argument("--num_cls", type=int, default=5, help="number of classes")
    parser.add_argument("--ckpt", default="model/BERT/best795.pth",type=str,help="Path to checkpoint")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="as the name")

    args = parser.parse_args()

    predict("i like it",args)