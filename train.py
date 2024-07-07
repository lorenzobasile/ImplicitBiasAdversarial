import torch
import argparse
from utils.data import get_dataloaders
from utils.training import train
from utils.models import get_model
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="resnet20", help="model architecture")
args = parser.parse_args()

model_name=args.model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size=64

if model_name=='resnet18' or model_name=='vit':
    dataset='imagenette'
else:
    dataset='cifar10'

dataloaders=get_dataloaders(dataset, batch_size, batch_size, shuffle_train=True, shuffle_test=False)
model=get_model(model_name)

if model_name=='resnet20':
    epochs=200
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
else:
    epochs=20
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                0.01,
                epochs=epochs,
                steps_per_epoch=len(dataloaders['train']),
                pct_start=0.1
            )

if not os.path.exists('trained_models/'+model_name):
    os.makedirs('trained_models/'+model_name)

print(f'Training {model_name} model')

model = model.to(device)
train(model, dataloaders, epochs, optimizer, scheduler)
torch.save(model.state_dict(), "trained_models/"+ model_name +"/clean.pt")
