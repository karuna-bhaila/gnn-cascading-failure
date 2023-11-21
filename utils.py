import copy
import numpy as np
import scipy

import torch


def data_split(data, train_ratio=0.60, val_ratio=0.20):
    train_mask = torch.zeros(len(data), dtype=torch.bool)
    val_mask = torch.zeros(len(data), dtype=torch.bool)
    test_mask = torch.zeros(len(data), dtype=torch.bool)

    perm = torch.randperm(len(data))

    num_train = round(len(data) * train_ratio)
    num_val = round(len(data) * val_ratio)

    train_mask[perm[:num_train]] = True
    val_mask[perm[num_train:num_train+num_val]] = True
    test_mask[perm[num_train+num_val:]] = True

    return data[train_mask], data[val_mask], data[test_mask]


def get_accuracy(pred, target):
    pred = pred.argmax(dim=1) if len(pred.size()) > 1 else pred
    target = target.argmax(dim=1) if len(target.size()) > 1 else target

    acc = (pred == target).sum().item() / target.numel()

    return acc * 100


# Source: https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
def loadmat(obj):
    '''
    this function should be called instead of direct scipy.loadmat
    as it cures thse problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''

    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], scipy.io.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, scipy.io.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    if isinstance(obj, str):
        data = scipy.io.loadmat(obj, struct_as_record=False, squeeze_me=True)
        return _check_keys(data)
    elif isinstance(obj, scipy.io.matlab.mat_struct):
        return _todict(obj)
    elif isinstance(obj, np.ndarray):
        return _tolist(obj)
    else:
        print("Invalid object passed.")
