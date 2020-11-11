import torch
import torch.nn as nn
import pdb
import os
import spacy
import json
import argparse
import torch.nn.functional as F
import torch.optim as optim
from dataset import FICDataset
from bert_classifier import Classifier
from transformers import AdamW, get_linear_schedule_with_warmup


def train(args, model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    for batch_idx, (input_ids, attention_mask, token_type_ids, cate) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        cate = cate.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids, cate)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input_ids), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def val(args, model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, token_type_ids, cate) in enumerate(val_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            cate = cate.to(device)
            outputs = model(input_ids, attention_mask, token_type_ids, cate)
            val_loss += outputs.loss.item() * args.batch_size
            if batch_idx % args.log_interval == 0:
                print('Validation: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(input_ids), len(val_loader.dataset),
                    100. * batch_idx / len(val_loader), outputs.loss.item()))

            probs = F.softmax(outputs.logits, dim=-1)
            pred = probs.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(cate.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    acc = correct / len(val_loader.dataset)

    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset), 100. * acc))
    return val_loss, acc


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, token_type_ids, cate) in enumerate(test_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            cate = cate.to(device)
            outputs = model(input_ids, attention_mask, token_type_ids, cate)
            test_loss += outputs.loss.item()  # sum up batch loss
            probs = F.softmax(outputs.logits)
            pred = probs.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(cate.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * acc))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Caption category classification.')
    parser.add_argument('--data_folder', type=str,
                        help='folder with data files saved by create_input_files.py.', default='')
    parser.add_argument('--model_folder', type=str,
                        help='base folder to save models.', default='')
    parser.add_argument('--batch-size', type=int, default=40, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--warmup_steps', type=int, default=50, metavar='N',
                        help='number of steps to warm up')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 0.5)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--split', type=str, default='test', metavar='SP',
                        help='split, train, val, or test')
    parser.add_argument('--train', action='store_true', default=True,
                        help='Train or Test.')
    parser.add_argument('--ckpt', type=str, default='', metavar='CK', help='checkpoint to load a model.')
    parser.add_argument('--category_file', type=str, default='', metavar='KA', help='file to load categories.')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},)

    with open(args.category_file, 'r') as f:
        categories = json.load(f)

    n_classes = len(categories)

    model = Classifier(n_classes=n_classes)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    if args.train:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        train_loader = torch.utils.data.DataLoader(FICDataset(args.data_folder, 'TRAIN'), **kwargs)
        val_loader = torch.utils.data.DataLoader(FICDataset(args.data_folder, "VAL"), **kwargs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=len(train_loader) * args.epochs)
        val_acc_best = 0
        for epoch in range(0, args.epochs):
            train(args, model, device, train_loader, optimizer, scheduler, epoch)
            val_loss, val_acc_cur = val(args, model, device, val_loader)
            if val_acc_cur > val_acc_best:
                val_acc_best = val_acc_cur
                model_file = os.path.join(args.model_folder, 'bert_classifier_best' + '.pt')
                torch.save(model.state_dict(), model_file)
            model_file = os.path.join(args.model_folder, 'bert_classifier' + '_' + str(epoch) + '.pt')
            torch.save(model.state_dict(), model_file)
    else:
        model.load_state_dict(torch.load(args.ckpt))
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False},
                      )
        test_loader = torch.utils.data.DataLoader(FICDataset(args.data_folder, 'TEST'), **kwargs)
        test(model, device, test_loader)


if __name__ == '__main__':
    main()

