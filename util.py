import os
import random
import numpy as np
import torch


def numpy2torch(array):
    """ Converts 3D numpy (H,W,C) ndarray to 3D PyTorch (C,H,W) tensor.
    :param:  array (numpy array)   : numpy array of shape (H, W, C)
    :return:  tensor (torch tensor)  : torch tensor of shape (C, H, W)
    """
    array = np.moveaxis(array, -1, 0)
    tensor = torch.from_numpy(array)
    return tensor


def torch2numpy(tensor):
    """ Converts 3D PyTorch (C,H,W) tensor to 3D numpy (H,W,C) ndarray.
    :param:  tensor (torch tensor)  : torch tensor of shape (C, H, W)
    :return:  array (numpy array)   : numpy array of shape (H, W, C)
    """
    tensor = tensor.to("cpu")
    array = tensor.detach().numpy()
    array = np.moveaxis(array, 0, -1)
    return array


def merge(a, b):
    """
    appends entries of dict b to a, works inplace modifying a
    :param:  a (dict)   : dict a
    :param:  b (dict)   : dict b
    :return:  a (dict)  : merged dict
    """
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key])
            else:
                a[key] += b[key]
        else:
            a[key] = b[key]
    return a


def set_manual_seed(seed: int = 1):
    """Set the seed for the PRNGs."""
    os.environ['PYTHONASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.benchmark = True


def get_baselines(datasets):
    """
    calculates the probability the most frequent class for each label of the datasets
    :param:  datasets (dict)            : dictionary of datasets
    :return:  label_freq (dict)         : Frequency of most frequent class for each label and probability of a randomly
     chosen class for each label
    """
    label_names = datasets['train'].dataset.labels
    for i, name in enumerate(label_names):
        if name == 'load_1' or name == 'load_2' or name == 'load_3':
            label_names[i] = 'load_obj'
    label_freq = {}
    for phase in ['train', 'val']:
        label_freq[phase] = {}
        dataset = datasets[phase]
        all_labels = np.empty((len(label_names), 0), int)
        for _, label in dataset:
            label = label.unsqueeze(dim=1)
            label = label.detach().numpy()
            all_labels = np.hstack((all_labels, label))
        label_freq[phase]['bal_acc'] = rand_choice(label_names)
        label_freq[phase]['acc'] = find_highest_frequency(all_labels, label_names)
    return label_freq


def get_baseline(dataset):
    """
    calculates the probability the most frequent class for each label of the dataset
    :param:  dataset (dataset)          : dataset which is analysed
    :return:  label_freq (dict)         : Frequency of most frequent class for each label and probability of a randomly chosen class for each label
    """
    label_names = dataset.labels
    for i, name in enumerate(label_names):
        if name == 'load_1' or name == 'load_2' or name == 'load_3':
            label_names[i] = 'load_obj'
    label_freq = {}
    all_labels = np.empty((len(label_names), 0), int)
    for _, label in dataset:
        label = label.unsqueeze(dim=1)
        label = label.detach().numpy()
        all_labels = np.hstack((all_labels, label))
    label_freq['bal_acc'] = rand_choice(label_names)
    label_freq['acc'] = find_highest_frequency(all_labels, label_names)
    return label_freq


def find_highest_frequency(arr, label_names):
    """
    calculates the probability the most frequent class for each label
    :param:  label_names (list of strings)      : list of label names
    :param:  arr (np array)                      : list of actual assigned label classes
    :return:  max_frq_for_label (list of float) : Frequency of most frequent class for each label
    """
    arr_for_label = {}
    max_frq_for_label = {}
    unique_label_names = list(set(label_names))
    for u_label in unique_label_names:
        max_frq_for_label[u_label] = 0
        arr_for_label[u_label] = np.empty(0, int)
    for label_name, col in zip(label_names, arr):
        arr_for_label[label_name] = np.hstack((arr_for_label[label_name], col))
    for label, value in arr_for_label.items():
        dict = {}
        for el in value:
            if str(el) in dict:
                dict[str(el)] += 1
            else:
                dict[str(el)] = 1
        max_frq_for_label[label] = max(dict.values()) / len(value)
    return max_frq_for_label


def rand_choice(label_names):
    """
    for each label the probability of a randomly chosen class is calculated
    :param:  label_names (list of strings)      : list of label names
    :return:  rand_choice (list of float)       : probability of a randomly chosen class for each label
    """
    if 'l_num' in label_names:
        classification_typ = 'class_specific'
    else:
        classification_typ = 'general'

    class_num = 22
    class_per_label = {
        'general': {
            'color': class_num, 'length': class_num, 'wall': class_num, 'roof': class_num, 'wheels': class_num,
            'load_obj': class_num, 'direction': class_num, 'load_1': class_num, 'load_2': class_num,
            'load_3': class_num},
        'class_specific': {
            'color': 6, 'length': 3, 'wall': 3, 'roof': 5, 'wheels': 3, 'l_num': 4, 'l_shape': 7, 'direction': 2,
        }
    }
    rand_choice = {}
    for label in label_names:
        rand_choice[label] = 1 / class_per_label[classification_typ][label]
    return rand_choice

