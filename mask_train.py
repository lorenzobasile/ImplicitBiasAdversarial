import foolbox
from utils.training import ess_train, adv_train
import torch
import argparse
from torch.utils.data import DataLoader
import os
from utils.data import get_dataloaders, AdversarialDataset
from utils.models import get_model
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--attack', type=str, default="FMN", help="attack type")
parser.add_argument('--model', type=str, default="resnet20", help="model architecture")
parser.add_argument('--mask', type=str, default="essential", help="type of mask (essential or adversarial)")
args = parser.parse_args()

save_figs=False
lam=0.01
model_name=args.model
if model_name=='resnet18':
    dataset='imagenette'
else:
    dataset='cifar10'
image_size=32 if dataset=='cifar10' else 224
attack=args.attack
batch_size=64
if args.mask=='essential':
    path="./essential/"+model_name+"/"
elif args.mask=='adversarial':
    path="./adversarial/"+attack+"/"+model_name+"/"
n_classes=10

if not os.path.exists(path):
    for i in range(n_classes):
        os.makedirs(path+"figures/"+str(i), exist_ok=True)
        os.makedirs(path+"masks/"+str(i), exist_ok=True)
        

dataloaders=get_dataloaders(dataset, batch_size, batch_size, shuffle_train=True, shuffle_test=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model=get_model(model_name)
base_model = base_model.to(device)
base_model.load_state_dict(torch.load("trained_models/"+model_name+"/clean.pt"))
base_model.eval()
fmodel = foolbox.models.PyTorchModel(base_model, bounds=(0,1))
if dataset=='cifar10':
    adv_dataloader=DataLoader(AdversarialDataset(fmodel, model_name, attack, dataloaders['test'], image_size, 'test'), batch_size=batch_size, shuffle=False)
elif dataset=='imagenette':
    adv_dataloader=DataLoader(AdversarialDataset(fmodel, model_name, attack, dataloaders['all'], image_size, 'all'), batch_size=batch_size, shuffle=False)
idx=0
for x, xadv, y in tqdm(adv_dataloader):
    if args.mask=='essential':
        idx=ess_train(base_model, x, y, lam, idx, path, image_size, save_figs)
    elif args.mask=='adversarial':
        idx=adv_train(base_model, x,  xadv, y, lam, idx, path, image_size, save_figs)

