from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
from lmdbdict import lmdbdict
from lmdbdict.methods import DUMPS_FUNC, LOADS_FUNC
import os
import numpy as np
import numpy.random as npr
import random
from functools import partial
import pdb
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import multiprocessing
import six


class Dataset(data.Dataset):

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, split, transform=None):
        self.opt = opt
        self.seq_per_img = 1

        if split == 'train':
            self.split = 'TRAIN'
        elif split == 'val':
            self.split = 'VAL'
        else:
            self.split = 'TEST'

        print('DataLoader loading files from: ', opt.data_folder)
        print('split: ', self.split)
        self.h = h5py.File(os.path.join(opt.data_folder, self.split + '_IMAGES' + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Load encoded captions (completely into memory)
        with open(os.path.join(opt.data_folder, self.split + '_CAPTIONS' + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(opt.data_folder, self.split + '_CAPLENS' + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load attributes and categories for evaluation (test) (completely into memory)
        with open(os.path.join(opt.data_folder, self.split + '_ATTRS' + '.json'), 'r') as j:
            self.attrs = json.load(j)

        with open(os.path.join(opt.data_folder, self.split + '_CATES' + '.json'), 'r') as j:
            self.cates = json.load(j)

        with open(os.path.join(opt.data_folder, 'WORDMAP.json'), 'r') as j:
            self.word_to_ix = json.load(j)

        self.ix_to_word = {u: v for v, u in self.word_to_ix.items()}
        self.transform = transform

        # load the json file which contains additional information about the dataset
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        """
        Setting input_label_h5 to none is used when only doing generation.
        For example, when you need to test on coco test set.
        """
        # if self.split != 'TEST':
        #     self.seq_length = len(self.captions[0]) - 2     # remove <s> and </s> to be consistent
        #     print('max sequence length in data is', self.seq_length)
        # else:
        #     self.seq_length = 1
        self.seq_length = len(self.captions[0]) - 2  # remove <s> and </s> to be consistent
        print('max sequence length in data is', self.seq_length)

        self.num_images = self.imgs.shape[0]
        print('read %d images.' % self.num_images)
        self.split_ix = [x for x in range(self.num_images)]

    def collate_func(self, batch):
        seq_per_img = self.seq_per_img
        img_batch = []
        label_batch = []
        infos = []
        gts = []
        wrapped = False

        for sample in batch:
            # fetch image
            img, caption, attr, cate, ix, it_pos_now, tmp_wrapped = sample
            if tmp_wrapped:
                wrapped = True

            img_batch.append(img)

            tmp_label = torch.zeros(seq_per_img, self.seq_length + 2, dtype=torch.int64)
            tmp_label[:, 1: self.seq_length + 1] = caption
            label_batch.append(tmp_label)

            gts.append(caption)

            # record associated info as well
            info_dict = {'ix': ix, 'id': self.split_ix[ix]}
            infos.append(info_dict)

        img_batch, label_batch, gts, infos = \
            zip(*sorted(zip(img_batch, label_batch, gts, infos), key=lambda x: 0, reverse=True))
        data = {'img_feats': torch.stack(img_batch)}

        data['labels'] = torch.stack(label_batch)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype='float32')
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch
        data['labels'] = data['labels'].reshape(len(batch), seq_per_img, -1)
        data['masks'] = data['masks'].reshape(len(batch), seq_per_img, -1)

        data['gts'] = gts  # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': it_pos_now,  # the it_pos_now of the last sample
                          'it_max': len(self.split_ix), 'wrapped': wrapped}
        data['infos'] = infos

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in data.items()}  # Turn all ndarray to torch tensor

        return data

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix, it_pos_now, wrapped = index
        img = torch.FloatTensor(self.imgs[ix] / 255.)

        if self.transform is not None:
            img = self.transform(img)

        caption = self.captions[ix]
        caplen = self.caplens[ix]
        caption = torch.LongTensor(caption[1: caplen-1] + [0] * (len(caption) - caplen))
        attr = torch.LongTensor(self.attrs[ix])
        cate = torch.LongTensor([self.cates[ix]])
        return img, caption, attr, cate, ix, it_pos_now, wrapped

    def __len__(self):
        return len(self.captions)


class DataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size

        # Initialize loaders and iters
        self.loaders, self.iters = {}, {}
        # Custom dataloaders
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for split in ['train', 'val', 'test']:
            self.dataset = Dataset(opt, split, transform=preprocess)
            if split == 'train':
                sampler = MySampler(self.dataset.split_ix, shuffle=True, wrap=True)
            else:
                sampler = MySampler(self.dataset.split_ix, shuffle=False, wrap=False)
            self.loaders[split] = data.DataLoader(dataset=self.dataset,
                                                  batch_size=self.batch_size,
                                                  sampler=sampler,
                                                  pin_memory=True,
                                                  num_workers=4,  # 4 is usually enough
                                                  collate_fn=partial(self.dataset.collate_func),
                                                  drop_last=False)
            self.iters[split] = iter(self.loaders[split])

    def get_batch(self, split):
        try:
            data = next(self.iters[split])
        except StopIteration:
            self.iters[split] = iter(self.loaders[split])
            data = next(self.iters[split])
        return data

    def reset_iterator(self, split):
        self.loaders[split].sampler._reset_iter()
        self.iters[split] = iter(self.loaders[split])

    def get_vocab_size(self):
        return self.dataset.get_vocab_size()

    @property
    def vocab_size(self):
        return self.get_vocab_size()

    def get_vocab(self):
        return self.dataset.get_vocab()

    def get_seq_length(self):
        return self.dataset.get_seq_length()

    @property
    def seq_length(self):
        return self.get_seq_length()

    def state_dict(self):
        def get_prefetch_num(split):
            if self.loaders[split].num_workers > 0:
                return (self.iters[split]._send_idx - self.iters[split]._rcvd_idx) * self.batch_size
            else:
                return 0

        return {split: loader.sampler.state_dict(get_prefetch_num(split)) for split, loader in self.loaders.items()}

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        for split in self.loaders.keys():
            self.loaders[split].sampler.load_state_dict(state_dict[split])


class MySampler(data.sampler.Sampler):
    def __init__(self, index_list, shuffle, wrap):
        self.index_list = index_list
        self.shuffle = shuffle
        self.wrap = wrap
        # if wrap, there will be not stop iteration called
        # wrap True used during training, and wrap False used during test.
        self._reset_iter()

    def __iter__(self):
        return self

    def __next__(self):
        wrapped = False
        if self.iter_counter == len(self._index_list):
            self._reset_iter()
            if self.wrap:
                wrapped = True
            else:
                raise StopIteration()
        if len(self._index_list) == 0:  # overflow when 0 samples
            return None
        elem = (self._index_list[self.iter_counter], self.iter_counter + 1, wrapped)
        self.iter_counter += 1
        return elem

    def next(self):
        return self.__next__()

    def _reset_iter(self):
        if self.shuffle:
            rand_perm = npr.permutation(len(self.index_list))
            self._index_list = [self.index_list[_] for _ in rand_perm]
        else:
            self._index_list = self.index_list

        self.iter_counter = 0

    def __len__(self):
        return len(self.index_list)

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        self._index_list = state_dict['index_list']
        self.iter_counter = state_dict['iter_counter']

    def state_dict(self, prefetched_num=None):
        prefetched_num = prefetched_num or 0
        return {
            'index_list': self._index_list,
            'iter_counter': self.iter_counter - prefetched_num
        }
