import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.models import MaskedClf, Map
import os

def train(model, dataloaders, n_epochs, optimizer, scheduler=None):

    '''
    Training routine for classification

    Param model (torch.nn.Module): classifier to be trained
    Param dataloaders (dictionary): dictionary containing a 'train' DataLoader and a 'test' DataLoader
    Param n_epochs (int): number of epochs
    Param optimizer (torch.optim.Optimizer): optimizer object
    Param scheduler (torch.optim.lr_scheduler._LRScheduler): optional learning rate scheduler
    '''

    loss=torch.nn.CrossEntropyLoss()
    device=torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    for epoch in range(n_epochs):   
        print("Epoch: ", epoch+1, '/', n_epochs)
        model.train()
        for x, y in dataloaders['train']:
            x=x.to(device)
            y=y.to(device)
            out=model(x)
            l=loss(out, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        model.eval()
        for i in ['train', 'test']:
            correct=0
            with torch.no_grad():
                for x, y in dataloaders[i]:
                    out=model(x.to(device))
                    correct+=(torch.argmax(out, axis=1)==y.to(device)).sum().item()
            print("Accuracy on "+i+" set: ", correct/len(dataloaders[i].dataset))


def ess_train(base_model, clean, y, lam, idx, path, img_size, figures=False):

    '''
    Function to train essential frequency maps

    Param base_model (torch.nn.Module): pre-trained classifier
    Param clean (torch.tensor): tensor containing a batch of clean, non-adversarial images
    Param y (torch.tensor): tensor containing a batch of image labels
    Param lam (float): parameter governing l_1 regularization
    Param idx (int): index of current image to be processed
    Param path (str): path to the folder used to store maps
    Param img_size (int): height (== width) of the images
    Param figures (bool): True if maps are to be stored after training, False otherwise

    Return: index of next image to be processed
    '''

    loss=torch.nn.CrossEntropyLoss()
    device=torch.device("cuda" if next(base_model.parameters()).is_cuda else "cpu")
    n=len(y)
    y=y.to(device)
    clean=clean.to(device)
    base_out=base_model(clean)
    losses=[[] for i in range(n)]
    #we consider only correctly classified images
    werecorrect=(np.where((torch.argmax(base_out, axis=1)==y).cpu())[0])
    for i in range(n):
        if not os.path.isfile(path+"maps/"+str(y[i].item())+"/"+str(idx)+".npy"):

            if i not in werecorrect:
                idx+=1
            else:  
                model=MaskedClf(Map((3, img_size, img_size)).to(device), base_model)
                for p in model.clf.parameters():
                    p.requires_grad=False
                model.map.train()
                optimizer=torch.optim.Adam(model.map.parameters(), lr=0.01)
                epoch=0
                while True:
                    out=model(clean[i])
                    l=loss(out, y[i].reshape(1))
                    penalty=model.map.M.abs().sum()
                    l+=penalty*lam
                    losses[i].append(l.item())
                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
                    #map entries are in [0,1]
                    model.map.M.data.clamp_(0.,1.)
                    epoch+=1
                    #train until convergence, for no less than 500 epochs and no more than 5000 epochs
                    if (epoch>500 and abs(l.item()-np.mean(losses[i][-20:]))<1e-5) or epoch>5000:
                        correct=torch.argmax(out, axis=1)==y[i]
                        if correct:
                            map=torch.fft.fftshift(model.map.M.detach().cpu())
                            map=map.squeeze().numpy()
                            if figures:
                                plt.axis('off')
                                plt.figure(figsize=(30,20))
                                plt.plot(losses[i])
                                plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"loss.png")
                                plt.close()
                                plt.figure()
                                plt.imshow(np.transpose(map, (1,2,0)))
                                plt.xticks([], [])
                                plt.yticks([], [])
                                plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+".svg", dpi=200, bbox_inches='tight', format='svg')
                                plt.close()
                                img=clean[i].reshape(3,img_size,img_size).detach().cpu().permute(1,2,0).numpy()
                                img_recon=model.map(clean[i]).reshape(3,img_size,img_size).detach().cpu().permute(1,2,0).numpy()
                                plt.figure()           
                                plt.imshow(img)
                                plt.xticks([], [])
                                plt.yticks([], [])
                                plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"clean.svg", dpi=200, bbox_inches='tight', format='svg')
                                plt.close()
                                plt.figure()
                                plt.imshow(np.log(torch.fft.fftshift(torch.fft.fft2(clean[i])).abs().reshape(3,img_size,img_size)[0].detach().cpu().numpy()/img_size/img_size), cmap='Blues')
                                plt.xticks([], [])
                                plt.yticks([], [])
                                plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"f_clean.svg", dpi=200, bbox_inches='tight', format='svg')
                                plt.close()
                                plt.figure()
                                plt.imshow(img_recon)
                                plt.xticks([], [])
                                plt.yticks([], [])
                                plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"clean_recon.svg", dpi=200, bbox_inches='tight', format='svg')
                                plt.close()
                            
                            np.save(path+"maps/"+str(y[i].item())+"/"+str(idx)+".npy", map)
                            del map
                        idx+=1
                        break
        else:
            idx+=1
    return idx

def adv_train(base_model, clean, adv, y, lam, idx, path, img_size, figures=False):

    '''
    Function to train adversarial frequency maps

    Param base_model (torch.nn.Module): pre-trained classifier
    Param clean (torch.tensor): tensor containing a batch of clean, non-adversarial images
    Param adv (torch.tensor): tensor containing a batch of adversarial images
    Param y (torch.tensor): tensor containing a batch of non-adversarial image labels
    Param lam (float): parameter governing l_1 regularization
    Param idx (int): index of current image to be processed
    Param path (str): path to the folder used to store maps
    Param img_size (int): height (== width) of the images
    Param figures (bool): True if maps are to be stored after training, false otherwise

    Return: index of next image to be processed
    '''

    loss=torch.nn.CrossEntropyLoss()
    device=torch.device("cuda" if next(base_model.parameters()).is_cuda else "cpu")
    n=len(clean)
    adv=adv.to(device)
    y=y.to(device)
    clean=clean.to(device)
    base_out=base_model(clean)
    base_adv=base_model(adv)
    losses=[[] for i in range(n)]
    wrong_labels=torch.argmax(base_adv, axis=1)
    #we consider only correctly classified and successfully attacked images
    wereadv=(np.where(torch.logical_and((torch.argmax(base_out, axis=1)==y).cpu(), (torch.argmax(base_adv, axis=1)!=y).cpu()))[0])
    for i in range(n):
        if not os.path.isfile(path+"maps/"+str(y[i].item())+"/"+str(idx)+".npy"):
            if i not in wereadv:
                idx+=1
            else:    
                model=MaskedClf(Map((3, img_size, img_size)).to(device), base_model)
                for p in model.clf.parameters():
                    p.requires_grad=False
                model.map.train()
                optimizer=torch.optim.Adam(model.map.parameters(), lr=0.01)
                epoch=0
                while True:
                    out=model(adv[i])
                    l=loss(out, wrong_labels[i].reshape(1))
                    penalty=model.map.M.abs().sum()
                    l+=penalty*lam
                    losses[i].append(l.item())
                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
                    #map entries are in [0,1]
                    model.map.M.data.clamp_(0., 1.)
                    epoch+=1
                    #train until convergence, for no less than 500 epochs and no more than 5000 epochs
                    if(epoch>500 and abs(l.item()-np.mean(losses[i][-20:]))<1e-5) or epoch>5000:
                        model.eval()
                        correct = torch.argmax(out, axis=1)==wrong_labels[i]
                        if correct:
                            map=torch.fft.fftshift(model.map.M.detach().cpu())
                            map=map.squeeze().numpy()
                            if figures:
                                plt.figure(figsize=(30,20))
                                plt.plot(losses[i])
                                plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"loss.png")
                                plt.close()
                                plt.figure()
                                plt.imshow(np.transpose(map, (1,2,0)))
                                plt.xticks([], [])
                                plt.yticks([], [])
                                plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"_"+str(wrong_labels[i].item())+".svg", dpi=200, bbox_inches='tight', format='svg')
                                plt.close()
                                adv_img=adv[i].reshape(3,img_size,img_size).detach().cpu().permute(1,2,0).numpy()
                                recon=model.map(adv[i]).reshape(3,img_size,img_size).detach().cpu().permute(1,2,0).numpy()
                                plt.figure()
                                plt.imshow(adv_img)
                                plt.xticks([], [])
                                plt.yticks([], [])
                                plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"adv.svg", dpi=200, bbox_inches='tight', format='svg')
                                plt.close()
                                plt.figure()
                                plt.imshow(np.log(torch.fft.fftshift(torch.fft.fft2(adv[i])).abs().reshape(3,img_size,img_size)[0].detach().cpu().numpy()/img_size/img_size), cmap='Blues')
                                plt.xticks([], [])
                                plt.yticks([], [])
                                plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"f_adv.svg", dpi=200, bbox_inches='tight', format='svg')
                                plt.close()                 
                                plt.figure()
                                plt.imshow(recon)
                                plt.xticks([], [])
                                plt.yticks([], [])
                                plt.savefig(path+"figures/"+str(y[i].item())+"/"+str(idx)+"pert_recon.svg", dpi=200, bbox_inches='tight', format='svg')   
                                plt.close()                  
                            np.save(path+"maps/"+str(y[i].item())+"/"+str(idx)+".npy", map)
                            del map
                            
                        idx+=1
                        break
        else:
            idx+=1
    return idx
