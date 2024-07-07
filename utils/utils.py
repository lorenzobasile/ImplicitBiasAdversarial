import torch

def cat(l):
    return torch.cat(l, axis=1)
def shuffle(data):
    return data[torch.randperm(len(data))]
def standardize(data):
    return (data-data.mean(0))/(data.std(0)+1e-6)
def normalize(data):
    if len(data.shape)>1 and data.shape[1]>1:
        return torch.nn.functional.normalize(data)
    else:
        return data
