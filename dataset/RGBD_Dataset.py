from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class RGBD_Dataset(Dataset):
    """
    Args:
        csv_path (string): Path to the csv file with annotations.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, csv_path, input_channels=3, transform=None):
        self.input_channels = input_channels
        csv_path = os.path.expanduser(csv_path)
        assert os.path.exists(csv_path), '%s not found' % csv_path

        # read csv file
        self.df = pd.read_csv(csv_path, header=0, names=['rgb_image_path', 'dep_image_path', 'cls_name'])
        self.df.cls_name, _ = pd.factorize(self.df.cls_name, sort=True)

        self._num_of_classes = len(self.df.cls_name.drop_duplicates())
        self._len_of_dataset = len(self.df.cls_name)
        self._transform = transform

    def get_num_of_classes(self):
        return self._num_of_classes

    def __len__(self):
        return self._len_of_dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        rgb_image_path, dep_image_path, cls_id = self.df.iloc[idx]
        # print(rgb_image_path)
        image = Image.open(rgb_image_path)
        if self.input_channels == 4:
            rgb_image = np.asarray(image)
            dep_image = np.load(dep_image_path)
            dep_image = np.expand_dims(dep_image, axis=-1)
            image = np.concatenate((rgb_image, dep_image), axis=-1)
        if self._transform is not None:
            image = self._transform(image)
        # There is no need to transfer into One-hot encoding
        # label = torch.zeros(self._num_of_classes, dtype=torch.long).scatter_(0, torch.from_numpy(np.array(cls_id)), 1)
        # label = (np.arange(self._num_of_classes) == cls_id).astype(np.float32)
        return image, cls_id


if __name__ == '__main__':
    """ code below are used for debugging.
    """
    from dataset.RGBD_transforms import Resize, RandomHorizontalFlip
    train_transform = transforms.Compose([
        Resize(224),
        RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # valid_transform = transforms.Compose([
    #     Resize(224),
    #     transforms.ToTensor(),
    # ])
    _train_transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # valid_transform = transforms.Compose([
    #     transforms.Resize(224),
    #     # transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    # ])

    train_dataset = RGBD_Dataset('~/vggface3d_sm/train.csv', input_channels=3, transform=train_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=0)
    import time

    start = time.time()
    for i, (image, label) in enumerate(train_dataloader):
        print(i, image.shape, label)
        if i >= 1:
            break
    elapsed = (time.time() - start)
    print("elapsed time: %d" % elapsed)
