import os
import numpy as np
from PIL import Image
import torch


def get_dataset(train_root, test_root, train_file_list, test_file_list, transform_train=None, transform_val=None):

    train_dataset = Dataset(train_root, train_file_list, transform=transform_train)
    test_dataset = Dataset(test_root, test_file_list, transform=transform_val)
    
    return train_dataset, test_dataset


def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(path).convert('RGB')
            return img
    except IOError:
        print('Cannot load image ' + path)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader):

        self.root = root
        self.transform = transform
        self.loader = loader

        image_list = []
        label_list = []

        with open(file_list) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_path, label_name = info.split(' ')
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]

        img = self.loader(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_list)