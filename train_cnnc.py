import pdb
import argparse
import numpy as np
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models.cnnc import CNNC
import opts
from dataset import *
from ficeval import *
from utils import *

# Model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


def mask_generator(caplens, max_len):
    mask = torch.ByteTensor(caplens.shape[0], max_len).zero_()
    for i in range(caplens.shape[0]):
        mask[i, :caplens[i].item()] = 1
    return mask


def main(opt):
    """
    Training and validation.
    """
    start_epoch = opt.start_epoch
    epochs_since_improvement = opt.epochs_since_improvement
    best_cider = 0
    word_map_file = opt.word_map_file
    with open(word_map_file, 'r') as f:
        word_map = json.load(f)

    # Initialize / load checkpoint
    if len(opt.checkpoint) == 0:
        # Convcap model
        model = CNNC(len(word_map), opt.num_layers, is_attention=opt.attention)
        optimizer = optim.RMSprop(model.parameters(), lr=opt.learning_rate)
    else:
        checkpoint = torch.load(opt.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_cider = checkpoint['cider']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to GPU, if available
    model = model.to(device)

    # using multiple GPUs, if available
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), " GPUs!")
        model = nn.DataParallel(model)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_loader = torch.utils.data.DataLoader(
        FICDataset(opt.data_folder, 'TRAIN', transform=transforms.Compose([preprocess])),
        batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        FICDataset(opt.data_folder, 'VAL', transform=transforms.Compose([preprocess])),
        batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, opt.epochs):
        # Decay learning rate if there is no improvement for 5 consecutive epochs, and terminate training after 20
        if opt.epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train(opt=opt, train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer,
              epoch=epoch)

        # One epoch's validation
        eval_result = validate(opt=opt, val_loader=val_loader, model=model, criterion=criterion, vocab=word_map)

        recent_cider = eval_result['CIDEr']
        # Check if there was an improvement
        is_best = recent_cider > best_cider
        best_cider = max(recent_cider, best_cider)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        to_save = 'vanilla'
        model_folder = os.path.join(opt.model_folder, to_save)
        if not os.path.exists(opt.model_folder):
            os.mkdir(opt.model_folder)
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        save_checkpoint(model_folder, epoch, epochs_since_improvement, model, optimizer, best_cider, is_best)


def train(opt, train_loader, model, criterion, optimizer, epoch):
    feat_h = 7
    feat_w = 7
    model.train()  # train mode (dropout and batchnorm is used)

    # One epoch of train
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        if i > 2000:
            break
        imgs = imgs.cuda()
        caps = caps.cuda()
        batch_size, max_tokens = caps.size()
        mask = mask_generator(caplens, caps.shape[1])
        optimizer.zero_grad()

        wordact, attn = model(imgs, caps)
        attn = attn.view(caps.shape[0], caps.shape[1], feat_h, feat_w)

        wordact = wordact[:, :, :-1]
        wordclass_v = caps[:, 1:]
        mask = mask[:, 1:].contiguous()

        wordact_t = wordact.permute(0, 2, 1).contiguous().view(batch_size * (max_tokens - 1), -1)
        wordclass_t = wordclass_v.contiguous().view(batch_size * (max_tokens - 1), 1)

        maskids = torch.nonzero(mask.view(-1), as_tuple=False).numpy().reshape(-1)

        # Cross-entropy loss and attention loss of Show, Attend and Tell
        loss = criterion(wordact_t[maskids, ...], wordclass_t[maskids, ...].contiguous().view(maskids.shape[0])) + (
                   torch.sum(torch.pow(1. - torch.sum(attn, 1), 2))) / (batch_size * feat_h * feat_w)
        # loss = criterion(wordact_t, wordclass_t[:, 0]) + \
        #        (torch.sum(torch.pow(1. - torch.sum(attn, 1), 2))) / (batch_size * feat_h * feat_w)

        loss.backward()
        optimizer.step()

        # Print status
        if i % opt.print_freq == 0 and i != 0:
            print('Epoch: [{}][{}/{}]\t Loss {:.4f}'.format(epoch, i, len(train_loader), loss))


def validate(opt, val_loader, model, criterion, vocab):
    model.eval()  # eval mode (no dropout or batchnorm)
    references = []  # references (true captions) for calculating BLEU-4 score
    hypotheses = []  # hypotheses (predictions)

    id_2_word = {x: y for y, x in vocab.items()}

    # validation
    with torch.no_grad():
        # batches
        for i, (imgs, caps, caplens) in enumerate(val_loader):
            if i > 200:
                break
            # pdb.set_trace()
            imgs = imgs.cuda()
            caps = caps.cuda()
            batch_size, max_tokens = caps.size()

            wordclass_feed = np.zeros((batch_size, max_tokens), dtype='int64')
            wordclass_feed[:, 0] = caps[:, 0].cpu().numpy()
            out_caps = np.empty((batch_size, 0)).tolist()

            for j in range(max_tokens - 1):
                wordclass = torch.from_numpy(wordclass_feed).cuda()
                wordact, _ = model(imgs, wordclass)
                wordact = wordact[:, :, :-1]
                wordact_t = wordact.permute(0, 2, 1).contiguous().view(batch_size * (max_tokens - 1), -1)
                wordprobs = F.softmax(wordact_t, dim=-1).cpu().data.numpy()
                wordids = np.argmax(wordprobs, axis=-1)

                for k in range(batch_size):
                    word = id_2_word[wordids[j + k * (max_tokens - 1)]]
                    out_caps[k].append(word)
                    if j < max_tokens - 1:
                        wordclass_feed[k, j + 1] = wordids[j + k * (max_tokens - 1)]

            # References
            for j in range(caps.shape[0]):
                img_cap = caps[j].tolist()  # [[26056, 10, 25, 2414, 5672, 71, 5, 0, 0, 0, 0]]
                img_caption = [id_2_word[w] for w in img_cap if w not in {vocab['<start>'], vocab['<pad>'],
                                                                          vocab['<end>'], vocab['<unk>']}]
                sent = ' '.join(img_caption)
                references.append({'caption': sent})

            # Hypothesis
            for j, p in enumerate(out_caps):
                pre_caption = [w for w in p if w not in ['<start>', '<pad>', '<end>', '<unk>']]
                sent = ' '.join(pre_caption)
                hypotheses.append({'caption': sent})

            # Cross-entropy loss and not including attention loss of Show, Attend and Tell
            wordclass_v = caps[:, 1:]
            wordclass_t = wordclass_v.contiguous().view(batch_size * (max_tokens - 1), 1)
            # warning: this might not be the correct loss calculation
            loss = criterion(wordact_t, wordclass_t[:, 0])

            # Keep track of metrics
            if i % opt.print_freq == 0 and i != 0:
                print('Validation: [{}/{}]\t Loss Total {:.4f}'.format(i, len(val_loader), loss))

    assert len(references) == len(hypotheses)
    scorer = FICScorer()
    ids = [str(k) for k in range(len(hypotheses))]
    hypo = {}
    refe = {}
    for k in range(len(hypotheses)):
        hypo[str(k)] = [hypotheses[k]]
        refe[str(k)] = [references[k]]
    final_scores = scorer.score(refe, hypo, ids)
    cache_path = os.path.join('eval_result/', 'cache_' + 'cnnc' + '.json')

    json.dump(hypo, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    return final_scores


if __name__ == '__main__':
    opt = opts.parse_opt()
    main(opt)
