import os
import torchvision
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import foolbox as fb


def get_dataloaders(dataset, train_batch_size, test_batch_size, shuffle_train=False, shuffle_test=False):

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
    Param exist_ok (bool): if True adversarial attacks are not regenerated if they already exist

    Return: index of next image to be processed
    '''
    def __init__(self, model, model_name, attack, dataloader, image_size, exist_ok=True):
        c="./data/adv/"+attack+"/"+model_name+"/clean.pt"
        a="./data/adv/"+attack+"/"+model_name+"/adv.pt"
        l="./data/adv/"+attack+"/"+model_name+"/lbl.pt"
        if os.path.isfile(c) and os.path.isfile(a) and os.path.isfile(l) and exist_ok:
            self.clean_imgs=torch.load(c)
            self.adv_imgs=torch.load(a)
            self.labels=torch.load(l)
            return
        if not os.path.exists("./data/adv/"+attack+"/"+model_name):
            os.makedirs("./data/adv/"+attack+"/"+model_name)
        self.clean_imgs=torch.empty(0,3,image_size,image_size)
        self.adv_imgs=torch.empty(0,3,image_size,image_size)
        self.labels=torch.empty(0, dtype=torch.int64)

        device=model.device
        for k, (x, y) in enumerate(dataloader):
            x=x.to(device)
            y=y.to(device)

            if attack=='PGD':
                adversary = fb.attacks.PGD()
            elif attack=='FMN':
                adversary = fb.attacks.LInfFMNAttack()
            elif attack=='DF':
                adversary = fb.attacks.LinfDeepFoolAttack()
            if 'FMN' in attack or 'DF' in attack:
                x_adv, clipped, is_adv = adversary(model, x, y, epsilons=None)
            else:
                x_adv, clipped, is_adv = adversary(model, x, y, epsilons=0.01)
            self.clean_imgs=torch.cat((self.clean_imgs, x.detach().cpu()))
            self.adv_imgs=torch.cat((self.adv_imgs, x_adv.detach().cpu()))

            self.labels=torch.cat((self.labels, y.detach().cpu()))
            self.labels.type(torch.LongTensor)
            

        self.labels, indices = torch.sort(self.labels)
        self.clean_imgs = self.clean_imgs[indices]
        self.adv_imgs = self.adv_imgs[indices]
        torch.save(self.clean_imgs, c)
        torch.save(self.adv_imgs, a)
        torch.save(self.labels, l)
        
    def __len__(self):
        return len(self.clean_imgs)

    def __getitem__(self, idx):
        clean, adv, labels= self.clean_imgs[idx], self.adv_imgs[idx], self.labels[idx]
        return clean, adv, labels


class MaskDataset(Dataset):

    '''
    Class that inherits from torch.utils.data.Dataset to store Fourier masks

    Attribute masks (torch.tensor): tensor that contains Fourier masks
    Attribute mask_indices (torch.tensor): tensor that contains the index of the image corresponding to each mask
    Attribute labels (torch.tensor): tensor that contains original ground truth labels for the image corresponding to the mask

    '''

    def __init__(self, path, image_size):

        m=path+"masks.pt"
        l=path+"labels.pt"
        i=path+"mask_indices.pt"
        if os.path.isfile(m) and os.path.isfile(l) and os.path.isfile(i):
            self.masks=torch.load(m)
            self.labels=torch.load(l)
            self.mask_indices=torch.load(i)
            return
        
        self.masks=torch.empty(0,3,image_size,image_size)
        self.labels=torch.empty(0, dtype=torch.int64)
        self.mask_indices=torch.empty(0, dtype=torch.int64)


        for c in range(10):
            class_list=sorted(os.listdir(path+str(c)),key=lambda x: int(os.path.splitext(x)[0]))
            temp_masks=torch.empty(0,3,image_size,image_size)
            temp_labels=torch.empty(0, dtype=torch.int64)
            temp_mask_indices=torch.empty(0, dtype=torch.int64)
            for mask in class_list:
                temp_masks=torch.cat((temp_masks, torch.from_numpy(np.load(path+str(c)+"/"+mask)).unsqueeze(0)))
                temp_labels=torch.cat((temp_labels, torch.tensor(c).unsqueeze(0)))
                temp_mask_indices=torch.cat((temp_mask_indices, torch.tensor(int(mask[:-4])).unsqueeze(0)))
            self.masks=torch.cat((self.masks, temp_masks))
            self.labels=torch.cat((self.labels, temp_labels))
            self.mask_indices=torch.cat((self.mask_indices, temp_mask_indices))
        
        torch.save(self.masks, path+"masks.pt")
        torch.save(self.labels, path+"labels.pt")
        torch.save(self.mask_indices, path+"mask_indices.pt")

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        return self.masks[idx], self.labels[idx], self.mask_indices[idx]
