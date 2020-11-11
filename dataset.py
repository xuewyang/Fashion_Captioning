import h5py
import json
import os
import torch
import pdb
from torch.utils.data import Dataset


class FICDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, data_folder, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_5' + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_5' + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_5' + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load attributes and categories for evaluation (test) (completely into memory)
        with open(os.path.join(data_folder, self.split + '_ATTRS_5' + '.json'), 'r') as j:
            self.attrs = json.load(j)

        with open(os.path.join(data_folder, self.split + '_CATES_5' + '.json'), 'r') as j:
            self.cates = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i / 255.])
        # img = torch.FloatTensor(self.imgs[i])
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        # cate = torch.LongTensor(self.cates[i])
        # attr = torch.LongTensor(self.attrs[i])

        return img, caption, caplen     #, cate, attr

    def __len__(self):
        return self.dataset_size
