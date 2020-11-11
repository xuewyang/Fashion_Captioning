from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from random import seed, choice, sample
import h5py
import numpy as np
import random
import re
from nltk.parse.corenlp import CoreNLPParser
from scipy.misc import imread, imresize
from tqdm import tqdm

st = CoreNLPParser()
from nltk.tokenize import word_tokenize
import nltk
import scipy.sparse as sp


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
# from processing.categorizing import filter_titles
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None
from nlgeval import NLGEval

import os
import pdb
import json
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from six.moves import cPickle

# from .rewards import get_scores, get_self_cider_scores

bad_endings = ['with', 'in', 'on', 'of', 'a', 'at', 'to', 'for', 'an', 'this', 'his', 'her', 'that']
bad_endings += ['the']


def random_one(impaths, imcaps, imattrs, imcate, captions_per_image=1):
    got = False
    while not got:
        i = random.randint(0, len(impaths))
        try:
            img = imread(impaths[i])
            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(
                    captions_per_image - len(imcaps[i]))]  # choice(imcaps[i]): get one cap from imcaps[i]
                attrs = imattrs[i]  # choice(imattrs[i]): get one attr from imattrs[i]
            else:
                captions = sample(imcaps[i], k=captions_per_image)  # if k = len(imcaps[i]), sample works like re-order
                attrs = imattrs[i]  # if k = len(imattrs[i]), sample works like re-order
            got = True
            return img, captions, attrs, imcate[i]
        except:
            got = False


# intersecting color names, e.g., bright white -> bright_white for saving in folders
def intersect_names(names):
    # names should be a list
    length = len(names)

    if length == 1:
        return names[0]
    result = ''
    for i in range(length-1):
        result += names[i]
        result += '_'
    try:
        result += names[-1]
    except:
        pdb.set_trace()
    return result


def create_input_files(data_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=25):
    """
    Creates input files for training, validation, and test data.

    :param data_json_path: path of JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    # Read JSON
    with open(data_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for ii, img in enumerate(data):
        if len(img['images']) == 0 or len(img['comments']) == 0:
            continue
        captions = []
        for c in img['comments']:
            # Update word frequency
            tokens = c['phra'][0].split()
            word_freq.update(tokens)
            if len(tokens) <= max_len:
                captions.append(tokens)

        if len(captions) == 0:
            continue
        path = os.path.join(image_folder, str(img['id']), intersect_names(img['images'][0]['color'].replace('/', '').split()), '0.jpeg')

        if ii < len(data)/10*9:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif ii > len(data)/10*9.5:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        else:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)                  # 6673
    assert len(val_image_paths) == len(val_image_captions)                      # 832
    assert len(test_image_paths) == len(test_image_captions)                    # 833

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]       # len(word_freq) = 27929
    word_map = {k: v + 1 for v, k in enumerate(words)}                          # len(words) = 9486
    word_map['<unk>'] = len(word_map) + 1       # 9487
    word_map['<start>'] = len(word_map) + 1     # 9488
    word_map['<end>'] = len(word_map) + 1       # 9489
    word_map['<pad>'] = 0                       # 0

    # Create a base/root name for all output files
    base_filename = 'scg' + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'   # 'scg_5_cap_per_img_5_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image
            # Create dataset inside HDF5 file to store images
            images_n = []
            for i, path in enumerate(tqdm(impaths)):
                # Read images
                try:
                    img = imread(impaths[i])
                except FileNotFoundError:
                    continue
                images_n.append(img)
            images = h.create_dataset('images', (len(images_n), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, img in enumerate(tqdm(images_n)):
                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]  # choice(imcaps[i]): get one cap from imcaps[i]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)      # if k = len(imcaps[i]), sample works like re-order

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                # try:
                #     img = imread(impaths[i])
                # except FileNotFoundError:
                #     continue
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))   # [9488, 49, 35, 1, 38, 50, 35, 43, 1, 46, 44, 9489, 0, 0, 0, ...]

                    # Find caption lengths
                    c_len = len(c) + 2
                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def create_description_tokenized_files(data_json_path, image_folder, output_file):
    # Read JSON
    with open(data_json_path, 'r') as j:
        data = json.load(j)
    data_n = []
    for img in data:
        img_n = {}
        if len(img['images']) == 0:
            continue
        if img['description'] is None:
            continue
        path = os.path.join(image_folder, str(img['id']),
                            intersect_names(img['images'][0]['color'].replace('/', '').split()), '0.jpeg')
        if not os.path.exists(path):
            continue
        if len(img['images']) == 0 or len(img['description']) == 0:
            continue
        # 1. tokenize sentence
        sent = list(st.tokenize(img['description']))
        # 2. lowercase sentences
        img_n['description'] = " ".join(x.lower() for x in sent)
        img_n['id'] = img['id']
        img_n['images'] = img['images']
        data_n.append(img_n)
    with open(output_file, 'w') as f:
        json.dump(data_n, fp=f, indent=4, ensure_ascii=False)


def create_description_input_files(data_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=50):
    """
    Creates input files for training, validation, and test data.

    :param data_json_path: path of JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    # Read JSON
    with open(data_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    train_image_attrs = []
    train_image_cate = []
    val_image_paths = []
    val_image_captions = []
    val_image_attrs = []
    val_image_cate = []
    test_image_paths = []
    test_image_captions = []
    test_image_attrs = []
    test_image_cate = []
    word_freq = Counter()

    for ii, img in enumerate(data):

        path = os.path.join(image_folder, str(img['id']), intersect_names(img['images'][0]['color'].replace('/', '').split()), '0.jpeg')
        if not os.path.exists(path):
            continue
        captions = []
        attrs = img['attrid']
        tokens = img['description'].split()
        tokens = [x.lower() for x in tokens]
        word_freq.update(tokens)

        if len(tokens) <= max_len:
            captions.append(tokens)

        if len(captions) == 0:
            continue

        if ii < len(data)/10*8:
            train_image_paths.append(path)
            train_image_captions.append(captions)
            train_image_attrs.append(attrs)
            train_image_cate.append(img['categoryid'])
        elif ii > len(data)/10*9:
            val_image_paths.append(path)
            val_image_captions.append(captions)
            val_image_attrs.append(attrs)
            val_image_cate.append(img['categoryid'])
        else:
            test_image_paths.append(path)
            test_image_captions.append(captions)
            test_image_attrs.append(attrs)
            test_image_cate.append(img['categoryid'])
    # Sanity check
    assert len(train_image_paths) == len(train_image_captions) == len(train_image_attrs) == len(train_image_cate)
    assert len(val_image_paths) == len(val_image_captions) == len(val_image_attrs) == len(val_image_cate)
    assert len(test_image_paths) == len(test_image_captions) == len(test_image_attrs) == len(test_image_cate)
    print("# of training data: ", len(train_image_paths))
    print("# of val data: ", len(val_image_paths))
    print("# of test data: ", len(test_image_paths))

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]       # len(word_freq) = 27929
    word_map = {k: v + 1 for v, k in enumerate(words)}                          # len(words) = 9486
    word_map['<unk>'] = len(word_map) + 1       # 9487
    word_map['<start>'] = len(word_map) + 1     # 9488
    word_map['<end>'] = len(word_map) + 1       # 9489
    word_map['<pad>'] = 0                       # 0

    # Create a base/root name for all output files
    base_filename = 'fc' + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'   # 'fc_5_cap_per_img_5_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'DESC_WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Save word freq to a JSON
    word_freq = dict(word_freq)
    word_freq_sorted = {k: v for k, v in sorted(word_freq.items(), key=lambda item: item[1])}
    with open(os.path.join(output_folder, 'DESC_WORDFREQ_' + base_filename + '.json'), 'w') as j:
        json.dump(word_freq_sorted, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, imattrs, imcate, split in [(train_image_paths, train_image_captions, train_image_attrs, train_image_cate, 'TRAIN'),
                                                    (val_image_paths, val_image_captions, val_image_attrs, val_image_cate, 'VAL'),
                                                    (test_image_paths, test_image_captions, test_image_attrs, test_image_cate, 'TEST')]:
        with h5py.File(os.path.join(output_folder, split + '_DESC_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image
            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            enc_attrs = []
            enc_cate = []
            caplens = []

            for i, img in enumerate(tqdm(impaths)):
                # Read images
                try:
                    img = imread(impaths[i])
                    # Sample captions
                    if len(imcaps[i]) < captions_per_image:
                        captions = imcaps[i] + [choice(imcaps[i]) for _ in range(
                            captions_per_image - len(imcaps[i]))]  # choice(imcaps[i]): get one cap from imcaps[i]
                        attrs = imattrs[i]  # choice(imattrs[i]): get one attr from imattrs[i]
                    else:
                        captions = sample(imcaps[i], k=captions_per_image)  # if k = len(imcaps[i]), sample works like re-order
                        attrs = imattrs[i]  # if k = len(imattrs[i]), sample works like re-order
                    enc_cate.append(imcate[i])
                except:
                    img = imread(impaths[i-1])
                    # Sample captions
                    if len(imcaps[i-1]) < captions_per_image:
                        captions = imcaps[i-1] + [choice(imcaps[i-1]) for _ in range(
                            captions_per_image - len(imcaps[i-1]))]  # choice(imcaps[i]): get one cap from imcaps[i]
                        attrs = imattrs[i-1]  # choice(imattrs[i]): get one attr from imattrs[i]
                    else:
                        captions = sample(imcaps[i-1],
                                          k=captions_per_image)  # if k = len(imcaps[i]), sample works like re-order
                        attrs = imattrs[i-1]  # if k = len(imattrs[i]), sample works like re-order
                    enc_cate.append([imcate[i-1]])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img
                # Sanity check
                assert len(captions) == captions_per_image
                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                    c_len = len(c) + 2
                    enc_captions.append(enc_c)
                    caplens.append(c_len)
                enc_attrs.append(attrs)


            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_DESC_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_DESC_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

            with open(os.path.join(output_folder, split + '_DESC_ATTRS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_attrs, j)

            with open(os.path.join(output_folder, split + '_DESC_CATES_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_cate, j)


def create_description_all_views_input_files(data_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=50):
    """
    Creates input files for training, validation, and test data.

    :param data_json_path: path of JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    # Read JSON
    with open(data_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    train_image_attrs = []
    train_image_cate = []
    val_image_paths = []
    val_image_captions = []
    val_image_attrs = []
    val_image_cate = []
    test_image_paths = []
    test_image_captions = []
    test_image_attrs = []
    test_image_cate = []
    word_freq = Counter()

    for ii, img in enumerate(data):
        for ppp in img['images']:
            for kkk in ppp.keys():
                color_name = ppp['color'].replace('/', '').split()
                if len(color_name) == 0:
                    continue
                path = os.path.join(image_folder, str(img['id']), intersect_names(color_name), kkk + '.jpeg')

                if not os.path.exists(path):
                    continue
                captions = []
                attrs = img['attrid']
                tokens = img['description'].split()
                tokens = [x.lower() for x in tokens]
                word_freq.update(tokens)

                if len(tokens) <= max_len:
                    captions.append(tokens)

                if len(captions) == 0:
                    continue

                if ii < len(data)/10*8:
                    train_image_paths.append(path)
                    train_image_captions.append(captions)
                    train_image_attrs.append(attrs)
                    train_image_cate.append(img['categoryid'])
                elif ii > len(data)/10*9:
                    val_image_paths.append(path)
                    val_image_captions.append(captions)
                    val_image_attrs.append(attrs)
                    val_image_cate.append(img['categoryid'])
                else:
                    test_image_paths.append(path)
                    test_image_captions.append(captions)
                    test_image_attrs.append(attrs)
                    test_image_cate.append(img['categoryid'])
    # Sanity check
    assert len(train_image_paths) == len(train_image_captions) == len(train_image_attrs) == len(train_image_cate)
    assert len(val_image_paths) == len(val_image_captions) == len(val_image_attrs) == len(val_image_cate)
    assert len(test_image_paths) == len(test_image_captions) == len(test_image_attrs) == len(test_image_cate)
    print("# of training data: ", len(train_image_paths))
    print("# of val data: ", len(val_image_paths))
    print("# of test data: ", len(test_image_paths))

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]       # len(word_freq) = 27929
    pdb.set_trace()
    word_map = {k: v + 1 for v, k in enumerate(words)}                          # len(words) = 9486
    word_map['<unk>'] = len(word_map) + 1       # 9487
    word_map['<start>'] = len(word_map) + 1     # 9488
    word_map['<end>'] = len(word_map) + 1       # 9489
    word_map['<pad>'] = 0                       # 0

    # Create a base/root name for all output files
    base_filename = 'fc' + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'   # 'fc_5_cap_per_img_5_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'DESC_WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Save word freq to a JSON
    word_freq = dict(word_freq)
    word_freq_sorted = {k: v for k, v in sorted(word_freq.items(), key=lambda item: item[1])}
    with open(os.path.join(output_folder, 'DESC_WORDFREQ_' + base_filename + '.json'), 'w') as j:
        json.dump(word_freq_sorted, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, imattrs, imcate, split in [(train_image_paths, train_image_captions, train_image_attrs, train_image_cate, 'TRAIN'),
                                                    (val_image_paths, val_image_captions, val_image_attrs, val_image_cate, 'VAL'),
                                                    (test_image_paths, test_image_captions, test_image_attrs, test_image_cate, 'TEST')]:
        with h5py.File(os.path.join(output_folder, split + '_DESC_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image
            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            enc_attrs = []
            enc_cate = []
            caplens = []

            for i, img in enumerate(tqdm(impaths)):
                # if i < 116730:
                #     continue
                # Read images
                try:
                    img = imread(impaths[i])
                    # Sample captions
                    if len(imcaps[i]) < captions_per_image:
                        captions = imcaps[i] + [choice(imcaps[i]) for _ in range(
                            captions_per_image - len(imcaps[i]))]  # choice(imcaps[i]): get one cap from imcaps[i]
                        attrs = imattrs[i]  # choice(imattrs[i]): get one attr from imattrs[i]
                    else:
                        captions = sample(imcaps[i], k=captions_per_image)  # if k = len(imcaps[i]), sample works like re-order
                        attrs = imattrs[i]  # if k = len(imattrs[i]), sample works like re-order
                    enc_cate.append(imcate[i])
                except:
                    img, captions, attrs, cate = random_one(impaths, imcaps, imattrs, imcate)
                    enc_cate.append([cate])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img
                # Sanity check
                assert len(captions) == captions_per_image
                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                    c_len = len(c) + 2
                    enc_captions.append(enc_c)
                    caplens.append(c_len)
                enc_attrs.append(attrs)


            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_DESC_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_DESC_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

            with open(os.path.join(output_folder, split + '_DESC_ATTRS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_attrs, j)

            with open(os.path.join(output_folder, split + '_DESC_CATES_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_cate, j)



def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.
    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1
    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue
        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(model_folder, epoch, epochs_since_improvement, model, optimizer, cider, is_best):
    """
    Saves model checkpoint.
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in cider score
    :param model: model
    :param optimizer: optimizer to update weights
    :param cider: validation cider score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    if torch.cuda.device_count() > 1:
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'cider': cider,
                 'model': model.module.state_dict(),    # save model.module for > 1 gpus
                 'optimizer': optimizer     # or use .state_dict()?
                 }
    else:
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'cider': cider,
                 'model': model.state_dict(),
                 'optimizer': optimizer
                 }
    filename = 'checkpoint_' + str(epoch) + '.pth.tar'
    torch.save(state, os.path.join(model_folder, filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        filename = 'checkpoint.pth.tar'
        torch.save(state, os.path.join(model_folder, 'BEST_' + filename))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.batch_sizes.size(0)
    _, ind = scores.data.topk(k, 1, True, True)
    correct = ind.eq(targets.data.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def break_down_description(desc):
    """
    Break description into several lines to show them
    :param desc: description
    :return:
    """
    words = desc.split()
    len_s = len(words)
    words_n = []
    for i in range(len_s):
        if i == int(len_s / 2.0):
            words_n.append('\n')
        words_n.append(words[i])

    desc_n = " ".join(x for x in words_n)
    return desc_n


def random_meta_data(file):
    with open(file, 'r') as f:
        data = json.load(f)
    random.seed(30)
    random.shuffle(data)
    random.seed(50)
    random.shuffle(data)
    file_name = '/home/xuewyang/Xuewen/Research/data/FACAD/jsons/meta_random_130254.json'
    with open(file_name, 'w') as f:
        json.dump(data, fp=f, indent=4, ensure_ascii=False)


json_file = '/home/xuewyang/Xuewen/Research/data/FACAD/jsons/meta_130254.json'
random_meta_data(json_file)


def get_pos_combinations(tokens):
    """
    tokens: [('Lounge', 'NN'), ('in', 'IN'), ('the', 'DT'), ('lap', 'NN'), ('of', 'IN'), ('luxury', 'NN'),
    ('with', 'IN'), ('this', 'DT'), ('short', 'JJ'), ('cashmere', 'NN'), ('robe', 'NN'), ('featuring', 'VBG'),
    ('large', 'JJ'), ('patch', 'NN'), ('pockets—so', 'VBD'), ('even', 'RB'), ('your', 'PRP$'), ('hands', 'NNS'),
    ('can', 'MD'), ('experience', 'VB'), ('superlative', 'JJ'), ('comfort', 'NN'), ('.', '.')]
    return: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    """
    pos_tags = nltk.pos_tag(tokens)
    attr_tokens = set()
    prev = None     # previous pos
    result = []
    first = ['JJ', 'JJR', 'JJS', 'VBD', 'VBG', 'VBN', 'NN', 'NNP', 'NNPS', 'NNS']
    second = ['NN', 'NNP', 'NNPS', 'NNS']
    for i, pos in enumerate(pos_tags):
        if prev in first and pos[1] in second:
            if len(result) != 0:
                result.pop()
            result.append(1)
            result.append(1)
            attr_tokens.add(tokens[i-1])
            attr_tokens.add(tokens[i])
        else:
            result.append(0)
        prev = pos[1]

    return result, attr_tokens


# text = word_tokenize("A plunge neck provides a dramatic update for this sleek and shimmery long-sleeve bodysuit.")
# print(get_pos_combinations(text))

def calculate_semantic_loss(scores, targets, pretrained, criterion_se):
    values, indices = torch.max(scores, dim=-1)
    output = pretrained(indices, 1)
    semantic_loss = criterion_se(output, targets.reshape(indices.shape[0]))
    # student = F.log_softmax(output1, 1)
    # output2 = pretrained(targets, 1)
    # teacher = F.softmax(output2, 1)
    # semantic_loss = criterion_se(student, teacher)
    return semantic_loss

# def calculate_semantic_loss(h, targets, pretrained, criterion_se):
#     values, indices = torch.max(scores, dim=-1)
#     output = pretrained(indices, 1)
#     new_scores = F.softmax(output, 1)
#     semantic_loss = criterion_se(new_scores, targets.reshape(indices.shape[0]))
#     # student = F.log_softmax(output1, 1)
#     # output2 = pretrained(targets, 1)
#     # teacher = F.softmax(output2, 1)
#     # semantic_loss = criterion_se(student, teacher)
#     return semantic_loss


def encode_onehot(labels):
    pdb.set_trace()
    classes = set(labels)

    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path):
    """Load citation network dataset (cora only for now)"""
    pdb.set_trace()

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# def accuracy(output, labels):
#     preds = output.max(1)[1].type_as(labels)
#     correct = preds.eq(labels).double()
#     correct = correct.sum()
#     return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_file(data_json_path, new_json_file):
    # Read JSON
    with open(data_json_path, 'r') as j:
        data = json.load(j)
    new_data = []
    for ii, img in enumerate(data):
        if len(img['images']) == 0:
            continue
        else:
            if img['detail_info'] is not None and len(img['detail_info']) != 0:
                try:        # \([^)]*\)
                    tokens = word_tokenize(re.sub(r'[^A-Za-z0-9]+', ' ', img['detail_info'].split('\n')[1]))
                except IndexError:
                    continue
            else:
                continue
        if len(img['description']) != 0:
            tokens = word_tokenize(re.sub(r'[^A-Za-z0-9]+', ' ', img['description']))

        tokens = [x.lower() for x in tokens]
        tokens = [wordnet_lemmatizer.lemmatize(x, pos='n') for x in tokens]
        tokens = [filter_titles(x) for x in tokens]
        img['description'] = ' '.join(tokens)
        title_tokens = word_tokenize(re.sub(r'[^A-Za-z0-9]+', ' ', img['title']))
        title_tokens = [x.lower() for x in title_tokens]
        title_tokens = [wordnet_lemmatizer.lemmatize(x, pos='n') for x in title_tokens]
        title_tokens = [filter_titles(x) for x in title_tokens]
        img['title'] = ' '.join(title_tokens)
        new_data.append(img)
    print(len(new_data))
    with open(new_json_file, 'w') as f:
        json.dump(new_data, fp=f, indent=4, ensure_ascii=False)

# new_json_file = '/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/jsons/meta_random_94026_preprocess.json'
# data_json_path='/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/jsons/meta_random_94026.json'
# preprocess_file(data_json_path, new_json_file)


def correct_file():
    file = os.path.join('/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/hdf5_all_view', 'VAL_DESC_CATES_fc_1_cap_per_img_0_min_word_freq' + '.json')
    with open(file, 'r') as j:
        cates = json.load(j)
        new_cates = []
        for cc in cates:
            if type(cc) is list:
                new_cates.append(cc[0])
            else:
                new_cates.append(cc)

    new_file = os.path.join('/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/hdf5_all_view', 'VAL_DESC_CATES_fc_1_cap_per_img_0_min_word_freq2' + '.json')
    with open(new_file, 'w') as f:
        json.dump(new_cates, f)

# correct_file()

def count_cate_attr():
    # count categories and attributes for all files
    cate_dict = {}
    attr_dict = {}
    file_c_train = '/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/hdf5_all_view/TRAIN_DESC_CATES_fc_1_cap_per_img_0_min_word_freq.json'
    file_a_train = '/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/hdf5_all_view/TRAIN_DESC_ATTRS_fc_1_cap_per_img_0_min_word_freq.json'
    file_c_test = '/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/hdf5_all_view/TEST_DESC_CATES_fc_1_cap_per_img_0_min_word_freq.json'
    file_a_test = '/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/hdf5_all_view/TEST_DESC_ATTRS_fc_1_cap_per_img_0_min_word_freq.json'
    file_c_val = '/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/hdf5_all_view/VAL_DESC_CATES_fc_1_cap_per_img_0_min_word_freq.json'
    file_a_val = '/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/hdf5_all_view/VAL_DESC_ATTRS_fc_1_cap_per_img_0_min_word_freq.json'
    with open(file_c_train, 'r') as f:
        cate_train = json.load(f)
        for cc in cate_train:
            if cc in cate_dict.keys():
                cate_dict[cc] += 1
            else:
                cate_dict[cc] = 0
    with open(file_c_test, 'r') as f:
        cate_test = json.load(f)
        for cc in cate_test:
            if cc in cate_dict.keys():
                cate_dict[cc] += 1
            else:
                cate_dict[cc] = 0
    with open(file_c_val, 'r') as f:
        cate_val = json.load(f)
        for cc in cate_val:
            if cc in cate_dict.keys():
                cate_dict[cc] += 1
            else:
                cate_dict[cc] = 0
    with open(file_a_train, 'r') as f:
        attr_train = json.load(f)
        for aa in attr_train:
            for cc in aa:
                # pdb.set_trace()
                if cc in attr_dict.keys():
                    attr_dict[cc] += 1
                else:
                    attr_dict[cc] = 0
    with open(file_a_test, 'r') as f:
        attr_test = json.load(f)
        for aa in attr_test:
            for cc in aa:
                if cc in attr_dict.keys():
                    attr_dict[cc] += 1
                else:
                    attr_dict[cc] = 0
    with open(file_a_val, 'r') as f:
        attr_val = json.load(f)
        for aa in attr_val:
            for cc in aa:
                if cc in attr_dict.keys():
                    attr_dict[cc] += 1
                else:
                    attr_dict[cc] = 0
    attr_dict_sorted = {k: v for k, v in sorted(attr_dict.items(), key=lambda item: item[1], reverse=True)}
    cate_dict_sorted = {k: v for k, v in sorted(cate_dict.items(), key=lambda item: item[1], reverse=True)}

    counter_file_count = '/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/jsons/category_count_93752.json'
    with open(counter_file_count, 'r') as f:
        categories = json.load(f)

    categories = categories.keys()
    cate_ids = {}
    for i, cate in enumerate(categories):
        cate_ids[cate] = i
    ids_cate = {value: key for key, value in cate_ids.items()}

    attr_file_count = '/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/jsons/attr_count_93752.json'
    with open(attr_file_count, 'r') as f:
        attrs = json.load(f)

    attrs_ids = {}
    for i, attr in enumerate(attrs):
        attrs_ids[attr] = i + 1
    ids_attrs = {value: key for key, value in attrs_ids.items()}

    new_att_dict = {}
    for att in attr_dict_sorted.keys():
        if att == 0:
            continue
        att_out = ids_attrs[att]
        if att_out in cate_ids:
            continue
        new_att_dict[ids_attrs[att]] = attr_dict_sorted[att]

    new_cate_dict = {}
    for cat in cate_dict_sorted.keys():
        new_cate_dict[ids_cate[cat]] = cate_dict_sorted[cat]
    pdb.set_trace()
# count_cate_attr()


def count_avg_cap_length():
    file_train = '/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/annotations/captions_train2014.json'
    file_val = '/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/annotations/captions_val2014.json'
    lengths = []
    with open(file_train, 'r') as f:
        train = json.load(f)
        for dd in train['annotations']:
            lengths.append(len(dd['caption'].split()))
    with open(file_val, 'r') as f:
        val = json.load(f)
        for dd in val['annotations']:
            lengths.append(len(dd['caption'].split()))
    pdb.set_trace()

    avg = sum(lengths) / len(lengths)

# count_avg_cap_length()


def get_perplexity(loss, round=2, base=2):
    if loss is None:
        return 0.
    return np.round(np.power(base, loss), round)


def calculate_metrics(references, hypotheses, vocab, nnl_loss):
    ppl = get_perplexity(nnl_loss.cpu().numpy())
    nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS',
                                       'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity',
                                       'GreedyMatchingScore'])
    id2word = {value: key for key, value in vocab.items()}
    hypos = []
    refers = []
    for hypo in hypotheses:
        hypo = [id2word[x] for x in hypo]
        hypo = ' '.join(hypo)
        hypos.append(hypo)
    for refer in references:
        refer = [id2word[x] for x in refer[0]]
        refer = ' '.join(refer)
        refers.append(refer)
    metrics_dict = nlgeval.compute_metrics(hyp_list=hypos, ref_list=[refers])
    return metrics_dict['Bleu_1'], metrics_dict['Bleu_2'], metrics_dict['Bleu_3'], metrics_dict['Bleu_4'], \
           metrics_dict['METEOR'], metrics_dict['ROUGE_L'], metrics_dict['CIDEr'], ppl


# def get_avg_attr(file):
#     data = np.load(file)
#
#     for dd in data:
#         pdb.set_trace()

# get_avg_attr('/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/hdf5_all_view/TRAIN_DESC_ATTRSOH_fc_1_cap_per_img_0_min_word_freq.npy')

def to_onehot(arr):
    onehot = np.zeros((1, 1098))
    for vv in arr:
        if vv != 0:
            onehot[0, vv-1] = 1
            # pdb.set_trace()

    return onehot


def count_perc(data_json_path):
    # Read JSON
    with open(data_json_path, 'r') as j:
        data = json.load(j)

    count1 = 0
    count2 = 0
    for ii, img in enumerate(data):
        # pdb.set_trace()
        if img['categoryid'] == 63 and (660 in img['attrid'] or 477 in img['attrid']):
            count1 += 1
        if img['categoryid'] == 63:
            count2 += 1
    print(count1 / count2)

# count_perc('/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/jsons/meta_all_93752.json')

def prepare_sup(file, new_file):
    with open(file, 'r') as j:
        data = json.load(j)
    new_data = []
    for dd in data:
        if dd['id'] < 500:
            new_data.append(dd)
    with open(new_file, 'w') as f:
        json.dump(new_data, fp=f, indent=4, ensure_ascii=False)

# f_all = '/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/jsons/meta_all_93752.json'
# f_some = '/home/xuewyang/Xuewen/Research/Fashion/Captioning/data/jsons/meta_500.json'
# prepare_sup(f_all, f_some)


def pickle_load(f):
    """ Load a pickle.
    Parameters
    ----------
    f: file-like object
    """
    if six.PY3:
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)


def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: pickled object
    f: file-like object
    """
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)


def if_use_feat(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc', 'newfc']:
        use_att, use_fc = False, True
    elif caption_model == 'language_model':
        use_att, use_fc = False, False
    elif caption_model in ['updown', 'topdown']:
        use_fc, use_att = True, True
    else:
        use_att, use_fc = True, False
    return use_fc, use_att


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[ix.item()]
            else:
                break

        if int(os.getenv('REMOVE_BAD_ENDINGS', '0')):
            flag = 0
            words = txt.split(' ')
            for j in range(len(words)):
                if words[-j - 1] not in bad_endings:
                    flag = -j
                    break
            txt = ' '.join(words[0:len(words) + flag])

        out.append(txt.replace('@@ ', ''))
    return out


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = (seq > 0).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class StructureLosses(nn.Module):
    """
    This loss is inspired by Classical Structured Prediction Losses for Sequence to Sequence Learning
    (Edunov et al., 2018).
    """

    def __init__(self, opt):
        super(StructureLosses, self).__init__()
        self.opt = opt
        self.loss_type = opt.structure_loss_type

    def forward(self, input, seq, data_gts):
        """
        Input is either logits or log softmax
        """
        out = {}

        batch_size = input.size(0)  # batch_size = sample_size * seq_per_img
        seq_per_img = batch_size // len(data_gts)

        assert seq_per_img == self.opt.train_sample_n, seq_per_img

        mask = (seq > 0).float()
        mask = torch.cat([mask.new_full((mask.size(0), 1), 1), mask[:, :-1]], 1)

        scores = get_scores(data_gts, seq, self.opt)
        if scores.shape[0] % seq_per_img == 0:
            scores = torch.from_numpy(scores).type_as(input).view(-1, seq_per_img)
        else:
            print(0)
            print(len(data_gts), len(seq))
            scores = torch.zeros((input.shape[0], 1)).type_as(input).view(-1, seq_per_img)
        out['reward'] = scores  # .mean()
        if self.opt.entropy_reward_weight > 0:
            entropy = - (F.softmax(input, dim=2) * F.log_softmax(input, dim=2)).sum(2).data
            entropy = (entropy * mask).sum(1) / mask.sum(1)
            print('entropy', entropy.mean().item())
            scores = scores + self.opt.entropy_reward_weight * entropy.view(-1, seq_per_img)
        # rescale cost to [0,1]
        costs = - scores
        if self.loss_type == 'risk' or self.loss_type == 'softmax_margin':
            costs = costs - costs.min(1, keepdim=True)[0]
            costs = costs / costs.max(1, keepdim=True)[0]
        # in principle
        # Only risk need such rescale
        # margin should be alright; Let's try.

        # Gather input: BxTxD -> BxT
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        if self.loss_type == 'seqnll':
            # input is logsoftmax
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            target = costs.min(1)[1]
            output = F.cross_entropy(input, target)
        elif self.loss_type == 'risk':
            # input is logsoftmax
            input = input * mask
            input = input.sum(1)
            input = input.view(-1, seq_per_img)

            output = (F.softmax(input.exp()) * costs).sum(1).mean()

            # test
            # avg_scores = input
            # probs = F.softmax(avg_scores.exp_())
            # loss = (probs * costs.type_as(probs)).sum() / input.size(0)
            # print(output.item(), loss.item())

        elif self.loss_type == 'max_margin':
            # input is logits
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input).max(1)[0] / 2
            output = output.mean()

        elif self.loss_type == 'multi_margin':
            # input is logits
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input)
            output = output.mean()

            # sanity test
            # avg_scores = input + costs
            # loss = F.multi_margin_loss(avg_scores, costs.min(1)[1], margin=0)
            # print(output, loss)

        elif self.loss_type == 'softmax_margin':
            # input is logsoftmax
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target)

        elif self.loss_type == 'real_softmax_margin':
            # input is logits
            # This is what originally defined in Kevin's paper
            # The result should be equivalent to softmax_margin
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target)

        elif self.loss_type == 'new_self_critical':
            """
            A different self critical
            Self critical uses greedy decoding score as baseline;
            This setting uses the average score of the rest samples as baseline
            (suppose c1...cn n samples, reward1 = score1 - 1/(n-1)(score2+..+scoren) )
            """
            baseline = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1] - 1)
            scores = scores - baseline
            # self cider used as reward to promote diversity (not working that much in this way)
            if getattr(self.opt, 'self_cider_reward_weight', 0) > 0:
                _scores = get_self_cider_scores(data_gts, seq, self.opt)
                _scores = torch.from_numpy(_scores).type_as(scores).view(-1, 1)
                _scores = _scores.expand_as(scores - 1)
                scores += self.opt.self_cider_reward_weight * _scores
            output = - input * mask * scores.view(-1, 1)
            output = torch.sum(output) / torch.sum(mask)

        out['loss'] = output
        return out


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        # Average over each token
        output = torch.sum(output) / torch.sum(mask)

        return output


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None

    def forward(self, input, target, mask):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)
        mask = mask.reshape(-1)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']


def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon,
                             weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon,
                          weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))


def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


class NoamOpt(object):
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


class ReduceLROnPlateau(object):
    "Optim wrapper that implements rate."

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, factor, patience, verbose, threshold,
                                                              threshold_mode, cooldown, min_lr, eps)
        self.optimizer = optimizer
        self.current_lr = get_lr(optimizer)

    def step(self):
        "Update parameters and rate"
        self.optimizer.step()

    def scheduler_step(self, val):
        self.scheduler.step(val)
        self.current_lr = get_lr(self.optimizer)

    def state_dict(self):
        return {'current_lr': self.current_lr,
                'scheduler_state_dict': self.scheduler.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        if 'current_lr' not in state_dict:
            # it's normal optimizer
            self.optimizer.load_state_dict(state_dict)
            set_lr(self.optimizer, self.current_lr)  # use the lr fromt the option
        else:
            # it's a schduler
            self.current_lr = state_dict['current_lr']
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            # current_lr is actually useless in this case

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


def get_std_opt(model, factor=1, warmup=2000):
    return NoamOpt(model.d_model, factor, warmup,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


# Function to rename multiple files
def rename_files(folder):
    for count, filename in enumerate(os.listdir(folder)):
        num, ext = filename.split('.')
        dst = str(int(num) + 893212) + '.' + ext        # 794075
        src = folder + filename
        dst = folder + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)


# folder = '/home/xuewyang/Xuewen/Research/data/FACAD/facad_vanilla_test_fc_2/'
# rename_files(folder)


def move_files(source_dir, target_dir):
    import shutil
    import os

    file_names = os.listdir(source_dir)

    for file_name in file_names:
        # try:
        shutil.move(os.path.join(source_dir, file_name), target_dir)
        # except:
        #     continue


# target_dir = '/home/xuewyang/Xuewen/Research/data/FACAD/facad_vanilla_train_fc/'
# move_files(folder, target_dir)


def merge_captions(train_caps_file, val_caps_file, test_caps_file, output_file):
    with open(train_caps_file, 'r') as f:
        train_caps = json.load(f)
    with open(val_caps_file, 'r') as f:
        val_caps = json.load(f)
    with open(test_caps_file, 'r') as f:
        test_caps = json.load(f)
    all_caps = train_caps + val_caps + test_caps

    with open(output_file, 'w') as f:
        json.dump(all_caps, f)


data_folder = '/home/xuewyang/Xuewen/Research/data/FACAD/jsons'
# train_caps_file_name = 'TRAIN_CAPTIONS_5.json'
# val_caps_file_name = 'VAL_CAPTIONS_5.json'
# test_caps_file_name = 'TEST_CAPTIONS_5.json'
# train_caps_file = os.path.join(data_folder, train_caps_file_name)
# val_caps_file = os.path.join(data_folder, val_caps_file_name)
# test_caps_file = os.path.join(data_folder, test_caps_file_name)
# output_file = '/home/xuewyang/Xuewen/Research/data/FACAD/jsons/CAPTIONS_5.json'
# merge_captions(train_caps_file, val_caps_file, test_caps_file, output_file)


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))


json_file = 'meta_129927_preprocess.json'


def get_pure_captions(data_folder, output_file):
    def tokens_to_caption(tokens, dic):
        to_avoid = [15804, 15805, 15806, 0]
        sent = ' '.join([dic[x] for x in tokens if x not in to_avoid])
        return sent

    caption_file = 'CAPTIONS_5.json'
    with open(os.path.join(data_folder, caption_file), 'r') as f:
        captions = json.load(f)
    wordmap_file = 'WORDMAP_5.json'
    with open(os.path.join(data_folder, wordmap_file), 'r') as f:
        word2idx = json.load(f)
    idx2word = {y: x for x, y in word2idx.items()}

    mode = 'a+'
    with open(os.path.join(data_folder, output_file), mode, encoding='utf-8') as f:
        for i, cap in enumerate(captions):
            # pdb.set_trace()
            sent = tokens_to_caption(cap, idx2word)
            cap_sent = {"id": i, "caption": sent}
            json_record = json.dumps(cap_sent, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(captions), os.path.join(data_folder, output_file)))


output_file = 'captions_facad.json'
# get_pure_captions(data_folder, output_file)


def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data


def shorten_label(json_file, output_file):
    '''
    For a label of size 32
    [15805, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15806, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    :return:
    '''
    # [15805, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15806, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    with open(json_file, 'r') as f:
        labels = json.load(f)

    labels_ = []
    for lab in labels:
        lab.remove(15805)
        lab.remove(15806)
        labels_.append(lab)

    pdb.set_trace()
    with open(output_file, 'w') as f:
        json.dump(labels_, f)


# json_file = os.path.join(data_folder, 'CAPTIONS_5.json')
# output_file = os.path.join(data_folder, 'CAPTIONS.json')
# shorten_label(json_file, output_file)

