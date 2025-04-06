
from utils.data import MapDataset
import argparse
import numpy as np
from utils.intrinsic_dimension import id_correlation
import torch
from anatome.similarity import svcca_distance


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

device='cuda' if torch.cuda.is_available() else 'cpu'

args = parser.parse_args()
image_size=224 if args.model=='resnet18' or args.model=='vit' else 32

dataset1=MapDataset(f'./adversarial/{args.attack}/{args.model}/maps/', image_size)
dataset2=MapDataset(f'./essential/{args.model}/maps/', image_size)

maps1=dataset1.maps
maps2=dataset2.maps

indices1=dataset1.map_indices.numpy()
indices2=dataset2.map_indices.numpy()

_, ind1, ind2= np.intersect1d(indices1, indices2, assume_unique=True, return_indices=True)

maps1=maps1[ind1]
maps2=maps2[ind2]

kernel_size=1 if image_size==32 else 7


m1=torch.nn.functional.max_pool2d(maps1, kernel_size=kernel_size)
m1=m1.reshape(-1, 3*m1.shape[-1]*m1.shape[-1])
m2=torch.nn.functional.max_pool2d(maps2, kernel_size=kernel_size)
m2=m2.reshape(-1, 3*m2.shape[-1]*m2.shape[-1])

print("IdCor: ", id_correlation(m1.to(device), m2.to(device), algorithm='twoNN', N=100))
print("SVCCA: ", 1-svcca_distance(m1.to(device), m2.to(device), accept_rate=0.99, backend='svd'))
print("Cosine sim: ", cosine_sim(m1.numpy(), m2.numpy()))