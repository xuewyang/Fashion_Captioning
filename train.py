import argparse
import sys
import pdb
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import opts
from dataset import *
from models.sat import SAT
from utils import *
from ficeval import *

# Model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


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
        model = SAT(attention_dim=opt.attention_dim, embed_dim=opt.emb_dim, decoder_dim=opt.decoder_dim,
                    vocab_size=len(word_map), dropout=opt.dropout)
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
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
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train(opt=opt, train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch)

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
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: loss layer
    :param optimizer: optimizer to update weights (if fine-tuning)
    :param epoch: epoch number
    """

    model.train()  # train mode (dropout and batchnorm is used)

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        scores, caps_sorted, decode_lengths, alphas, sort_ind = model(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]        # torch.Size([32, 52])

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        # Calculate loss
        loss = criterion(scores.data, targets.data)
        # Add doubly stochastic attention regularization
        loss += opt.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if opt.grad_clip is not None:
            clip_gradient(optimizer, opt.grad_clip)

        # Update weights
        optimizer.step()

        # Print status
        if i % opt.print_freq == 0 and i != 0:
            print('Epoch: [{}][{}/{}]\t Loss {:.4f}'.format(epoch, i, len(train_loader), loss))


def validate(opt, val_loader, model, criterion, vocab):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param model: model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    model.eval()  # eval mode (no dropout or batchnorm)
    references = []  # references (true captions) for calculating BLEU-4 score
    hypotheses = []  # hypotheses (predictions)

    id_2_word = {x: y for y, x in vocab.items()}

    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens) in enumerate(val_loader):
            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            scores, caps_sorted, decode_lengths, alphas, sort_ind = model(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores.data, targets.data)
            # Add doubly stochastic attention regularization
            loss += opt.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            if i % opt.print_freq == 0 and i != 0:
                print('Validation: [{}/{}]\t Loss Total {:.4f}'.format(i, len(val_loader), loss))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
            # References
            for j in range(caps_sorted.shape[0]):
                img_cap = caps_sorted[j].tolist()  # [[26056, 10, 25, 2414, 5672, 71, 5, 0, 0, 0, 0]]
                img_caption = [id_2_word[w] for w in img_cap if w not in {vocab['<start>'], vocab['<pad>'],
                                                                          vocab['<end>'], vocab['<unk>']}]
                sent = ' '.join(img_caption)
                # remove <start> and pads
                references.append({'caption': sent})

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)        # [100, 33, 26058], [100, 33]
            preds = preds.tolist()

            for j, p in enumerate(preds):
                pre_cap = p[:decode_lengths[j]]
                pre_caption = [id_2_word[w] for w in pre_cap if w not in {vocab['<start>'], vocab['<pad>'],
                                                                          vocab['<end>'], vocab['<unk>']}]
                sent = ' '.join(pre_caption)
                hypotheses.append({'caption': sent})
                # remove pads  decode_lengths: [33, 32, 32, 31, 31, 31, 29, 29, 28,]

    assert len(references) == len(hypotheses)
    scorer = FICScorer()
    ids = [str(k) for k in range(len(hypotheses))]
    hypo = {}
    refe = {}
    for k in range(len(hypotheses)):
        hypo[str(k)] = [hypotheses[k]]
        refe[str(k)] = [references[k]]
    final_scores = scorer.score(refe, hypo, ids)
    cache_path = os.path.join('eval_result/', 'cache_' + 'sat' + '.json')

    json.dump(hypo, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    return final_scores


if __name__ == '__main__':
    opt = opts.parse_opt()
    main(opt)
