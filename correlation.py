
from utils.data import MaskDataset
import argparse
import numpy as np
import scipy.stats
from tqdm import tqdm
from dadapy.data import Data


def compute_id(dataset):
    '''
    Function to compute Intrinsic Dimension of a data set

    Param dataset (np.array): data set

    Return: intrinsic dimension of data set, estimated with TwoNN (Facco et al., 2017)
    '''
    data = Data(dataset, maxk=3)
    del dataset
    id=data.compute_id_2NN()[0]
    return id

def cosine_sim(dataset1, dataset2):
    '''
    Function to compute cosine similarity between two data sets (in our case, M_EF and M_AF)

    Param dataset1 (np.array): first data set to correlate
    Param dataset2 (np.array): second data set to correlate

    Return: mean and standard deviation of cosine similarities
    '''
    product=np.sum(dataset1*dataset2, axis=1)
    norm1=np.linalg.norm(dataset1, 2, axis=1)
    norm2=np.linalg.norm(dataset2, 2, axis=1)
    corrs=np.divide(np.divide(product, norm1), norm2)
    return np.mean(corrs), np.std(corrs)

parser = argparse.ArgumentParser()

parser.add_argument('--attack', type=str, default="FMN", help="attack type")
parser.add_argument('--model', type=str, default="resnet20", help="model architecture")
parser.add_argument('--compute_individual', type=str, default='False')

args = parser.parse_args()
image_size=224 if args.model=='resnet18' else 32

dataset1=MaskDataset(f'./adversarial/{args.attack}/{args.model}/masks/', image_size)
dataset2=MaskDataset(f'./essential/{args.model}/masks/', image_size)

masks1=dataset1.masks.numpy()
masks2=dataset2.masks.numpy()

indices1=dataset1.mask_indices.numpy()
indices2=dataset2.mask_indices.numpy()

_, ind1, ind2= np.intersect1d(indices1, indices2, assume_unique=True, return_indices=True)

masks1=masks1[ind1].reshape(-1, 3*image_size*image_size)
masks2=masks2[ind2].reshape(-1, 3*image_size*image_size)

mu, sigma=cosine_sim(masks1, masks2)
print(f'Cosine similarity: mean {mu}, std {sigma}')

if args.compute_individual=='True':
    print(f'Id of second set: {compute_id(masks1)}')
    print(f'Id of second set: {compute_id(masks2)}')
    
full_data=np.concatenate([masks1, masks2], axis=1)
noshuffle=compute_id(full_data)
print(f'No-shuffle Id: {noshuffle}')

shuffle=[]
for _ in tqdm(range(50)):
    del full_data
    full_data=np.concatenate([np.random.permutation(masks1), masks2], axis=1)
    shuffle.append(compute_id(full_data))
shuffle=np.array(shuffle)
print(f'Shuffled Id mean: {shuffle.mean()}, std: {shuffle.std()}')
zscore=(noshuffle-shuffle.mean())/shuffle.std()
print(f'Z-score: {zscore}')
p=scipy.stats.norm.cdf(zscore)
print(f'P-value: {p}')
