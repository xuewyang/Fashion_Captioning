import argparse
import sys
import pdb
from sklearn.metrics import average_precision_score
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

# Data parameters

# Model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


def main(opt):
    """
    Training and validation.
    """
    word_map_file = opt.word_map_file
    with open(word_map_file, 'r') as f:
        word_map = json.load(f)

    # load checkpoint
    model = SAT(attention_dim=opt.attention_dim, embed_dim=opt.emb_dim, decoder_dim=opt.decoder_dim,
                vocab_size=len(word_map), dropout=opt.dropout)
    checkpoint = torch.load(opt.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model'])
    model.eval()

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

    test_loader = torch.utils.data.DataLoader(
        FICDataset(opt.data_folder, 'TEST', transform=transforms.Compose([preprocess])),
        batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)

    test_result = test(opt=opt, test_loader=test_loader, model=model, criterion=criterion, vocab=word_map)


def calculate_pr(hypo, refe, atts):
    hypo_ = hypo['caption'].split(' ')
    refe_ = refe['caption'].split(' ')
    positive = [x for x in refe_ if x in atts]      # all positive = tp + fn
    true_p = [x for x in hypo_ if x in positive]    # tp
    selected = [x for x in hypo_ if x in atts]      # tp + fp
    prec = len(true_p) / len(selected)
    reca = len(true_p) / len(positive)

    return prec, reca


def test(opt, test_loader, model, criterion, vocab):
    """
    Performs one epoch's validation.
    :param test_loader: DataLoader for validation data.
    :param model: model
    :param criterion: loss layer
    :return: metric evaluations
    """
    attribute_file = '/home/xuewyang/Xuewen/Research/data/FACAD/jsons/attr_count_129927.json'
    category_file = '/home/xuewyang/Xuewen/Research/data/FACAD/jsons/category_count_129927.json'
    with open(attribute_file, 'r') as f:
        attributes = json.load(f)
    with open(category_file, 'r') as f:
        categories = json.load(f)
    model.eval()  # eval mode (no dropout or batchnorm)
    references = []  # references (true captions) for calculating BLEU-4 score
    hypotheses = []  # hypotheses (predictions)

    id_2_word = {x: y for y, x in vocab.items()}

    loss_total = []  # total loss
    precisions = []
    recalls = []
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens) in enumerate(test_loader):
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
            loss_total.append(loss)

            # Keep track of metrics
            if i % opt.print_freq == 0 and i != 0:
                print('Test: [{}/{}]\t Loss Total {:.4f}'.format(i, len(test_loader), loss))

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
        pre, rec = calculate_pr(hypotheses[j], references[j], attributes)
        precisions.append(pre)
        recalls.append(rec)

    final_scores = scorer.score(refe, hypo, ids)
    cache_path = os.path.join('test_result/', 'test_cache_' + 'sat' + '.json')

    json.dump(hypo, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...
    precision_mean = sum(precisions) / len(precisions)
    recall_mean = sum(recalls) / len(recalls)
    final_scores['att_precision_mean'] = precision_mean
    final_scores['att_recall_mean'] = recall_mean
    metric_path = os.path.join('test_result/', 'test_metric_' + 'sat' + '.json')

    json.dump(final_scores, open(metric_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    return final_scores


if __name__ == '__main__':
    opt = opts.parse_opt()
    main(opt)
