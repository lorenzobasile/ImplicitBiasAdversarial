from utils.data import MaskDataset, compute_id
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


n=20
n_classes=10
for model in ['resnet20', 'cct']:
    attacks=['FMN', 'PGD', 'DF']
    pvalues=np.zeros((len(attacks),n,n_classes))
    zscores=np.zeros((len(attacks),n,n_classes))
    for a, attack in enumerate(attacks):
        print("Evaluating :", model, ", attacked by: ", attack)
        dataset1=MaskDataset(f'./essential/{model}/masks/', 32)
        dataset2=MaskDataset(f'./adversarial/{attack}/{model}/masks/', 32)
        masks1=dataset1.masks.numpy()
        masks2=dataset2.masks.numpy()
        indices1=dataset1.mask_indices.numpy()
        indices2=dataset2.mask_indices.numpy()
        labels1=dataset1.labels.numpy()
        labels2=dataset2.labels.numpy()
        _, ind1, ind2= np.intersect1d(indices1, indices2, assume_unique=True, return_indices=True)
        for i in range(n):
            random_order=np.random.permutation(10)
            for k in range(1,n_classes+1):
                kclasses1=ind1[np.in1d(labels1[ind1], random_order[:k])]
                kclasses2=ind2[np.in1d(labels2[ind2], random_order[:k])]
                m1=masks1[kclasses1].reshape(-1, 3*32*32)
                m2=masks2[kclasses2].reshape(-1, 3*32*32)         
                full_data=np.concatenate([m1, m2], axis=1)
                noshuffle=compute_id(full_data)
                shuffle=[]
                for _ in range(50):
                    full_data=np.concatenate([np.random.permutation(m1), m2], axis=1)
                    shuffle.append(compute_id(full_data))
                shuffle=np.array(shuffle)
                zscore=(noshuffle-np.mean(shuffle))/np.std(shuffle)
                p=scipy.stats.norm.cdf(zscore)
                pvalues[a,i,k-1]=p
                zscores[a,i,k-1]=zscore
                
    np.save(f'{model}_pvalues.npy', pvalues)
    np.save(f'{model}_zscores.npy', zscores)
    plt.figure(figsize=(12,6))
    plt.xlabel("$k$ (number of classes)", fontsize=20)
    plt.ylabel("P-value", fontsize=20)
    plt.xticks(range(n_classes+1))
    for a, attack in enumerate(attacks):
        plt.plot(range(1,n_classes+1),pvalues[a].mean(axis=0), 'o-',label=attack)
    plt.legend(fontsize=16)
    plt.savefig(f'{model}_class_specificity.png', dpi=300)
    plt.show()
