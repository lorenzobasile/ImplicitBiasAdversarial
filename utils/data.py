import os
import torchvision
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import foolbox as fb
from tqdm import trange, tqdm

def get_dataloaders(dataset, train_batch_size, test_batch_size, shuffle_train=False, shuffle_test=False, unnorm=False):

    '''
    Utility function to obtain DataLoaders

    Param dataset (str): name of the data set to load (either cifar10 or imagenette)
    Param train_batch_size (int): batch size for training set
    Param test_batch_size (int): batch size for test set
    Param shuffle_train (bool): if True the training set is shuffled
    Param shuffle_test (bool): if True the test set is shuffled
    
    Return: DataLoaders for chosen data set
    '''

    if dataset=='cifar10':
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(), 
                transforms.RandomCrop(32, 4), 
                transforms.ToTensor()
                ]),
            'test': transforms.Compose([
                transforms.ToTensor()
            ]),
        }
        if unnorm:
            data_transforms['train']=transforms.Compose([
                transforms.ToTensor()
            ])
        datasets = {x: torchvision.datasets.CIFAR10(root='./data', train=(x=='train'), download=True, transform=data_transforms[x]) for x in['train', 'test']}

        dataloaders = {'train': DataLoader(datasets['train'], batch_size=train_batch_size, shuffle=shuffle_train),
                   'test': DataLoader(datasets['test'], batch_size=test_batch_size, shuffle=shuffle_test)}
    elif dataset=='imagenette':
        data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                ])
        datasets = {'train': torchvision.datasets.ImageFolder('./data/imagenette2-320/train', data_transforms),
                   'test': torchvision.datasets.ImageFolder('./data/imagenette2-320/val', data_transforms),
                    'all': torch.utils.data.ConcatDataset([torchvision.datasets.ImageFolder('./data/imagenette2-320/val', data_transforms),
                                                            torchvision.datasets.ImageFolder('./data/imagenette2-320/train', data_transforms)
                                                            ])}    
        dataloaders= {'train': DataLoader(datasets['train'], batch_size=train_batch_size, shuffle=shuffle_train),
                   'test': DataLoader(datasets['test'], batch_size=test_batch_size, shuffle=shuffle_test),
                    'all': DataLoader(datasets['all'], batch_size=test_batch_size, shuffle=shuffle_test)}
    
    return dataloaders

class AdversarialDataset(Dataset):

    '''
    Class that inherits from torch.utils.data.Dataset to generate and store adversarial images
    Supported attacks: FMN, PGD, DF (DeepFool)

    Attribute clean_imgs (torch.tensor): tensor that contains original, clean images
    Attribute adv_imgs (torch.tensor): tensor that contains adversarial images
    Attribute labels (torch.tensor): tensor that contains original ground truth labels
    '''


    '''
    Constructor of AdversarialDataset: creates and stores adversarial examples

    Param model (foolbox.models.PyTorchModel): pre-trained classifier, converted to foolbox model
    Param model_name (str): name of the classifier
    Param attack (str): type of attack to employ (FMN, PGD, DF)
    Param dataloader (torch.data.DataLoader): dataloader containing images to attack
    Param image_size (int): height (== width) of the images
    Param fold (str): data type (train, test or all)
    Param exist_ok (bool): if True adversarial attacks are not regenerated if they already exist
    Param eps (float): perturbation magnitude for PGD attack

    Return: index of next image to be processed
    '''
    def __init__(self, model, model_name, attack, dataloader, image_size, fold='test', exist_ok=True, eps=8/255):
        c="./data/adv/"+attack+"/"+model_name+"/"+fold+"/clean.pt"
        a="./data/adv/"+attack+"/"+model_name+"/"+fold+"/adv.pt"
        l="./data/adv/"+attack+"/"+model_name+"/"+fold+"/lbl.pt"
        a_l="./data/adv/"+attack+"/"+model_name+"/"+fold+"/adv_lbl.pt"

        if os.path.isfile(c) and os.path.isfile(a) and os.path.isfile(l) and os.path.isfile(a_l) and exist_ok:
            self.clean_imgs=torch.load(c)
            self.adv_imgs=torch.load(a)
            self.labels=torch.load(l)
            self.adv_labels=torch.load(a_l)
            return
        if not os.path.exists("./data/adv/"+attack+"/"+model_name+"/"+fold):
            os.makedirs("./data/adv/"+attack+"/"+model_name+"/"+fold)
        self.clean_imgs=torch.empty(0,3,image_size,image_size)
        self.adv_imgs=torch.empty(0,3,image_size,image_size)
        self.labels=torch.empty(0, dtype=torch.int64)
        self.adv_labels=torch.empty(0, dtype=torch.int64)

        device=model.device
        for k, (x, y) in tqdm(enumerate(dataloader)):
            x=x.to(device)
            y=y.to(device)

            if 'PGD' in attack:
                adversary = fb.attacks.PGD()
            elif attack=='FMN':
                adversary = fb.attacks.LInfFMNAttack()
            elif attack=='DF':
                adversary = fb.attacks.LinfDeepFoolAttack()
            if 'FMN' in attack or 'DF' in attack:
                x_adv, clipped, is_adv = adversary(model, x, y, epsilons=None)
            else:
                x_adv, clipped, is_adv = adversary(model, x, y, epsilons=eps)
            self.clean_imgs=torch.cat((self.clean_imgs, x.detach().cpu()))
            self.adv_imgs=torch.cat((self.adv_imgs, x_adv.detach().cpu()))

            self.labels=torch.cat((self.labels, y.detach().cpu()))
            self.labels.type(torch.LongTensor)
            self.adv_labels=torch.cat((self.adv_labels, model(x_adv).argmax(-1).detach().cpu()))
            self.adv_labels.type(torch.LongTensor)
            

        self.labels, indices = torch.sort(self.labels)
        self.clean_imgs = self.clean_imgs[indices]
        self.adv_imgs = self.adv_imgs[indices]
        self.adv_labels = self.adv_labels[indices]
        torch.save(self.clean_imgs, c)
        torch.save(self.adv_imgs, a)
        torch.save(self.labels, l)
        torch.save(self.adv_labels, a_l)
        
    def __len__(self):
        return len(self.clean_imgs)

    def __getitem__(self, idx):
        clean, adv, labels, adv_labels = self.clean_imgs[idx], self.adv_imgs[idx], self.labels[idx], self.adv_labels[idx]
        return clean, adv, labels, adv_labels


class MapDataset(Dataset):

    '''
    Class that inherits from torch.utils.data.Dataset to store Fourier maps

    Attribute maps (torch.tensor): tensor that contains Fourier maps
    Attribute map_indices (torch.tensor): tensor that contains the index of the image corresponding to each map
    Attribute labels (torch.tensor): tensor that contains original ground truth labels for the image corresponding to the map

    '''

    def __init__(self, path, image_size):

        m=path+"maps.pt"
        l=path+"labels.pt"
        i=path+"map_indices.pt"
        if os.path.isfile(m) and os.path.isfile(l) and os.path.isfile(i):
            self.maps=torch.load(m)
            self.labels=torch.load(l)
            self.map_indices=torch.load(i)
            return

        maps=[]
        labels=[]
        map_indices=[]

        for c in trange(10):
            class_list=sorted(os.listdir(path+str(c)),key=lambda x: int(os.path.splitext(x)[0]))
            for map in class_list:
                m=torch.from_numpy(np.load(path+str(c)+"/"+map)).unsqueeze(0)
                maps.append(m)
                labels.append(torch.tensor(c).unsqueeze(0))
                map_indices.append(torch.tensor(int(map[:-4])).unsqueeze(0))
        self.maps=torch.cat(maps)
        self.labels=torch.cat(labels)
        self.map_indices=torch.cat(map_indices)
        
        torch.save(self.maps, path+"maps.pt")
        torch.save(self.labels, path+"labels.pt")
        torch.save(self.map_indices, path+"map_indices.pt")

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        return self.maps[idx], self.labels[idx], self.map_indices[idx]
