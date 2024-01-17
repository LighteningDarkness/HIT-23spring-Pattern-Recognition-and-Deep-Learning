import torch
from PIL import Image
from torchvision import transforms
import argparse
from MLP import MLP
def Predict(image,args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=MLP(args)
    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)
    model.eval()

    
    im = Image.open(image).convert('L')
    im=im.resize((28,28),resample=Image.Resampling.LANCZOS)

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((args.mean,),(args.std,))])

    im=transform(im)
    im=im.to(device)
    im=im.view(1,1,28,28)

    output=model(im)

    print(f"Prediction is {output.argmax(dim=1).item()}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Make predictions")
    parser.add_argument("--mean", type=float, default=0.1307, help="average")
    parser.add_argument("--std", type=float, default=0.3081, help="variance")  
    parser.add_argument("--ckpt", type=str, default="ckpt/best.pth", help="Directory for checkpoint")
    parser.add_argument("--dim_input", type=int, default=28*28, help="dim of input")
    parser.add_argument("--dim_hidden", type=int, default=1000, help="dim of hidden layer")
    parser.add_argument("--num_cls", type=int, default=10, help="number of classes")
    args = parser.parse_args()

    Predict('8.jpg',args)
