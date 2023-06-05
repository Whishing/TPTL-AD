import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset


class pre_dataset(Dataset):
    def __init__(self, root, empty=False, transform=None):
        super(pre_dataset, self).__init__()
        self.class_dir = os.listdir(root)
        self.data = []
        self.targets = []
        self.empty = empty
        self.transform = transform
        if not self.empty:
            for target, dir_name in enumerate(self.class_dir):
                img_path_list = os.listdir(os.path.join(root, dir_name))
                for img_path in img_path_list:
                    img = Image.open(os.path.join(root, dir_name, img_path)).convert("RGB").resize((256, 256))
                    self.data.append(np.array(img).reshape(1, 256, 256, 3))
                    self.targets.append(target)
            self.data = np.vstack(self.data)
        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class ft_dataset(Dataset):
    def __init__(self, root, empty=False, transform=None, pre_cl_num=1000):
        super(ft_dataset, self).__init__()
        self.class_dir = os.listdir(root)
        self.data = []
        self.targets = []
        self.empty = empty
        self.transform = transform
        self.pre_cl_num = pre_cl_num
        if not self.empty:
            for target, dir_name in enumerate(self.class_dir):
                img_path_list = os.listdir(os.path.join(root, dir_name))
                for img_path in img_path_list:
                    img = Image.open(os.path.join(root, dir_name, img_path)).convert("RGB").resize((256, 256))
                    self.data.append(np.array(img).reshape(1, 256, 256, 3))
                    self.targets.append(target + self.pre_cl_num)
            self.data = np.vstack(self.data)
        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
