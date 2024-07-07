import torch
from utils.data import get_dataloaders, AdversarialDataset
from utils.models import get_model, Mask, MaskedClf
from torch.utils.data import DataLoader
from utils.data import AdversarialDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import foolbox
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--attack', type=str, default="FMN", help="attack type")
args = parser.parse_args()

eps=8/255
train_adv=True
if 'PGD' in args.attack and len(args.attack)>3:
    eps = int(args.attack[4:])/255
    print(eps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model=get_model('resnet20')
base_model = base_model.to(device)
base_model.load_state_dict(torch.load("trained_models/resnet20/clean.pt"))
base_model.eval()
fmodel = foolbox.models.PyTorchModel(base_model, bounds=(0,1))
if not os.path.exists(f'class_specific/{args.attack}'):
    os.makedirs(f'class_specific/{args.attack}', exist_ok=True)

loss=torch.nn.CrossEntropyLoss()
dataloaders=get_dataloaders('cifar10', 128, 128, shuffle_train=True, shuffle_test=False, unnorm=True)
dataset=AdversarialDataset(fmodel, 'resnet20', args.attack, dataloaders['train'], 32, 'train', eps=eps)
dataset.adv_imgs=dataset.adv_imgs[dataset.adv_labels.argsort()]
dataset.labels=dataset.labels[dataset.adv_labels.argsort()]
dataset.clean_imgs=dataset.clean_imgs[dataset.adv_labels.argsort()]
dataset.adv_labels=dataset.adv_labels[dataset.adv_labels.argsort()]
for lbl in range(10):
    x = dataset.clean_imgs[dataset.labels==lbl].to(device)
    xadv = dataset.adv_imgs[dataset.adv_labels==lbl].to(device)
    y = dataset.labels[dataset.labels==lbl].to(device)
    yadv = dataset.adv_labels[dataset.adv_labels==lbl].to(device)
    xp = dataset.clean_imgs[dataset.adv_labels==lbl].to(device)
    yp = dataset.labels[dataset.adv_labels==lbl].to(device)
    losses=[]
    correct_adv=(base_model(xp).argmax(-1)==yp)
    xadv=xadv[correct_adv]
    yadv=yadv[correct_adv]
    yp = yp[correct_adv]
    xp = xp[correct_adv]
    correct=(base_model(x).argmax(-1)==y)
    y=y[correct]
    x=x[correct]
    model=MaskedClf(Mask((3, 32, 32)).to(device), base_model)
    for p in model.clf.parameters():
        p.requires_grad=False
    model.mask.train()
    optimizer=torch.optim.Adam(model.mask.parameters(), lr=0.01)
    for e in range(5000):
        print(e, end='\r')
        if train_adv:
            out=model(xadv)
            l=loss(out, yp)
            penalty=model.mask.M.abs().sum()
            l+=penalty*0.001
            avg=l
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            model.mask.M.data.clamp_(0., 1.)
        else:
            avg=0
        out=model(x)
        l=loss(out, y)
        penalty=model.mask.M.abs().sum()
        l+=penalty*0.001
        avg+=l
        avg/=2
        losses.append(avg.item())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        model.mask.M.data.clamp_(0., 1.)
        c=yadv[0].cpu().item()
        if(e>500 and abs(avg.item()-np.mean(losses[-20:]))<1e-3):
            mask=torch.fft.fftshift(model.mask.M.detach().cpu())
            mask=mask.squeeze().numpy()
            np.save(f'class_specific/{args.attack}/{c}.npy', mask)
            plt.figure()
            plt.imshow(np.transpose(mask, (1,2,0)))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.savefig(f"class_specific/{args.attack}/{c}.svg", dpi=200, bbox_inches='tight', format='svg')
            plt.close()
            break
    del model
adv_test_dataloader=DataLoader(AdversarialDataset(fmodel, 'resnet20', args.attack, dataloaders['test'], 32, 'test', eps=eps), batch_size=1, shuffle=False)
correct=0
adversarial=0
masked=0
for x,xadv,y,yadv in adv_test_dataloader:
    x=x.to(device)
    xadv=xadv.to(device)
    y=y.to(device)
    yadv=yadv.to(device)
    c=y[0].cpu().item()
    clean_out=base_model(x)
    correct_images=(clean_out.argmax(-1)==y)
    adv_out=base_model(xadv)
    correct+=correct_images.sum()
    if correct_images.sum()==0:
        continue
    masked_model=MaskedClf(Mask((3, 32, 32)).to(device), base_model)
    masked_model.mask.M.data=torch.fft.ifftshift(torch.tensor(np.load(f'class_specific/{args.attack}/{yadv.item()}.npy')))
    masked_model.mask=masked_model.mask.to(device)
    masked_model.eval()
    adversarial+=(masked_model(xadv).argmax(-1)==y).sum()
    masked_model=MaskedClf(Mask((3, 32, 32)).to(device), base_model)
    masked_model.mask.M.data=torch.fft.ifftshift(torch.tensor(np.load(f'class_specific/{args.attack}/{y.item()}.npy')))
    masked_model.mask=masked_model.mask.to(device)
    masked_model.eval()
    masked+=(masked_model(x).argmax(-1)==y).sum()
print("Correctly classified: ", correct.item(), "Correct after using the mask: ", masked, "Adversarial undone after using the mask:", adversarial)