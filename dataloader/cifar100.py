import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

SEED = int(23356)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

CLASS_NUMBER = 100

# LABEL = b"fine_labels"
LABEL = b"coarse_labels"

DATA_ROOT = 'cifar-100-python'


class DataSet(Dataset):

    def __init__(self, is_train, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.images = []
        self.labels = []

        if is_train:
            with open(DATA_ROOT+"/train", "rb") as fo:
                dicts = pickle.load(fo, encoding="bytes")
                for index in range(len(dicts[LABEL])):
                    img_r = np.reshape(dicts[b"data"][index][:1024], (32, 32))
                    img_g = np.reshape(dicts[b"data"][index][1024:2048], (32, 32))
                    img_b = np.reshape(dicts[b"data"][index][2048:], (32, 32))
                    img = np.stack([img_r, img_g, img_b], axis=-1)
                    img = img.transpose([2, 0, 1])
                    self.images.append(img)
                    self.labels.append(dicts[LABEL][index])
        else:
            with open(DATA_ROOT+"/test", "rb") as fo:
                dicts = pickle.load(fo, encoding="bytes")
                for index in range(len(dicts[LABEL])):
                    img_r = np.reshape(dicts[b"data"][index][:1024], (32, 32))
                    img_g = np.reshape(dicts[b"data"][index][1024:2048], (32, 32))
                    img_b = np.reshape(dicts[b"data"][index][2048:], (32, 32))
                    img = np.stack([img_r, img_g, img_b], axis=-1)
                    img = img.transpose([2, 0, 1])
                    self.images.append(img)
                    self.labels.append(dicts[LABEL][index])

    def __getitem__(self, item):
        data, label = self.images[item], self.labels[item]
        data = torch.Tensor(data)
        return data, label

    def __len__(self):
        return len(self.images)

