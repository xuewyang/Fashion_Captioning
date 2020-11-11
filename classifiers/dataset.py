import h5py
import json
import os
import torch
import pdb
from torch.utils.data import Dataset
from transformers import BertTokenizer


class FICDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, data_folder, split, max_len=50):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        self.max_len = max_len
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_5' + '.json'), 'r') as j:
            self.captions = json.load(j)

        with open(os.path.join(data_folder, 'WORDMAP_5.json'), 'r') as f:
            word_map = json.load(f)

        self.dictionary = {x: y for y, x in word_map.items()}

        # Load categories for evaluation (test) (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CATES_5' + '.json'), 'r') as j:
            self.cates = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        cate = torch.LongTensor([self.cates[i]])

        img_caption = [self.dictionary[w] for w in self.captions[i] if w not in [0, 15804, 15805, 15806]]
        sent = ' '.join(img_caption)
        # input_ids = self.tokenizer.encode(sent, add_special_tokens=True)
        # attention_mask = [1] * len(input_ids)
        # padding_length = self.max_len - len(input_ids)
        # input_ids = input_ids + [0] * padding_length
        # attention_mask = attention_mask + [0] * padding_length
        encoded_dict = self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=50, padding='max_length',
                                                  return_attention_mask=True, truncation=True)

        input_ids = torch.LongTensor(encoded_dict['input_ids'])
        attention_mask = torch.LongTensor(encoded_dict['attention_mask'])
        token_type_ids = torch.LongTensor(encoded_dict['token_type_ids'])

        return input_ids, attention_mask, token_type_ids, cate

    def __len__(self):
        return self.dataset_size
