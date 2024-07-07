from utils.data import MaskDataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib as mpl
import seaborn as sns
from tqdm import trange
from utils.intrinsic_dimension import id_correlation

device='cuda' if torch.cuda.is_available() else 'cpu'

n=20
n_classes=10
for model in ['resnet20']:
    attacks=['FMN', 'PGD', 'DF']
    pvalues=np.zeros((len(attacks),n,n_classes))
    correlation=np.zeros((len(attacks),n,n_classes))
    for a, attack in enumerate(attacks):
        print("Evaluating :", model, ", attacked by: ", attack)
        dataset1=MaskDataset(f'./essential/{model}/masks/', 32)
        dataset2=MaskDataset(f'./adversarial/{attack}/{model}/masks/', 32)
        masks1=dataset1.masks
        masks2=dataset2.masks
        indices1=dataset1.mask_indices.numpy()
        indices2=dataset2.mask_indices.numpy()
        labels1=dataset1.labels.numpy()
        labels2=dataset2.labels.numpy()
        _, ind1, ind2= np.intersect1d(indices1, indices2, assume_unique=True, return_indices=True)
        for i in trange(n):
            random_order=np.random.permutation(10)
            for k in range(1,n_classes+1):
                kclasses1=ind1[np.in1d(labels1[ind1], random_order[:k])]
                kclasses2=ind2[np.in1d(labels2[ind2], random_order[:k])]
                m1=masks1[kclasses1].reshape(-1, 3*32*32)
                m2=masks2[kclasses2].reshape(-1, 3*32*32)
                corr=id_correlation(m1.to(device), m2.to(device), algorithm='twoNN', N=100)      
                pvalues[a,i,k-1]=corr['p']
                correlation[a,i,k-1]=corr['corr']
    

    np.save(f'{model}_pvalues.npy', pvalues)
    np.save(f'{model}_correlation.npy', correlation)

    #pvalues=np.load(f'{model}_pvalues.npy')
    #correlation=np.load(f'{model}_correlation.npy')

    fig, ax1 = plt.subplots(figsize=(12,6))

    colors = mpl.colormaps['tab20'].colors

    ax1.set_xlabel("$k$ (number of classes)", fontsize=20)
    ax1.set_ylabel("P-value ", fontsize=20)
    for a, attack in enumerate(attacks):
        ax1.plot(range(1,n_classes+1),pvalues[a].mean(axis=0), 'o--', label=f'{attack}', c=colors[2*a+1])
    ax1.tick_params(axis='y')
    

    ax2 = ax1.twinx() 

    ax2.set_ylabel("$I_d$ Correlation", fontsize=20) 
    for a, attack in enumerate(attacks):
        plt.plot(range(1,n_classes+1),correlation[a].mean(axis=0), 'o-', label=f'{attack if "DF" not in attack else "DeepFool"}', c=colors[2*a])
    ax2.tick_params(axis='y')
    ax2.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, prop={'size': 20})

    fig.tight_layout() 
    plt.savefig(f'{model}_class_specificity.svg', dpi=200, bbox_inches='tight', format='svg')