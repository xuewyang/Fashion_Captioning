import pdb
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# Layers adapted for captioning from https://arxiv.org/abs/1705.03122
# def Conv1d(in_channels, out_channels, kernel_size, padding, dropout=0.):
#     m = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
#     std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * in_channels))
#     m.weight.data.normal_(mean=0, std=std)
#     m.bias.data.zero_()
#     return nn.utils.weight_norm(m)


# def Embedding(num_embeddings, embedding_dim):
#     m = nn.Embedding(num_embeddings, embedding_dim)
#     m.weight.data.normal_(0, 0.1)
#     return m


# def Linear(in_features, out_features, dropout=0.):
#     m = nn.Linear(in_features, out_features)
#     m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
#     m.bias.data.zero_()
#     return nn.utils.weight_norm(m)


# class Vgg16Feats(nn.Module):
#     def __init__(self):
#         super(Vgg16Feats, self).__init__()
#         vgg = models.vgg16(pretrained=True)
#         self.features_nopool = nn.Sequential(*list(vgg.features.children())[:-1])
#         self.features_pool = list(vgg.features.children())[-1]
#         self.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
#
#     def forward(self, x):
#         x = self.features_nopool(x)
#         x_pool = self.features_pool(x)
#         x_feat = x_pool.view(x_pool.size(0), -1)
#         y = self.classifier(x_feat)
#         return x_pool, y


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim):
        super(AttentionLayer, self).__init__()
        self.in_projection = nn.Linear(conv_channels, embed_dim)
        self.out_projection = nn.Linear(embed_dim, conv_channels)
        self.bmm = torch.bmm

    def forward(self, x, wordemb, imgsfeats):
        residual = x
        x = (self.in_projection(x) + wordemb) * math.sqrt(0.5)
        b, c, f_h, f_w = imgsfeats.size()
        y = imgsfeats.view(b, c, f_h * f_w)
        x = self.bmm(x, y)
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=-1)
        x = x.view(sz)
        attn_scores = x
        y = y.permute(0, 2, 1)
        x = self.bmm(x, y)
        s = y.size(1)
        x = x * (s * math.sqrt(1.0 / s))
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)

        return x, attn_scores


class CNNC(nn.Module):
    def __init__(self, num_wordclass, num_layers=1, is_attention=True, nfeats=512, dropout=.1):
        super(CNNC, self).__init__()
        self.vgg = models.vgg16(pretrained=True)        # finetune all layers
        # for p in self.vgg.parameters():
        #     p.requires_grad = False
        # self.fine_tune()    # finetune encoder
        self.features_nopool = nn.Sequential(*list(self.vgg.features.children())[:-1])
        self.features_pool = list(self.vgg.features.children())[-1]
        self.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1])

        self.nimgfeats = 4096
        self.is_attention = is_attention
        self.nfeats = nfeats
        self.dropout = dropout
        self.emb_0 = nn.Embedding(num_wordclass, nfeats)
        self.emb_1 = nn.Linear(nfeats, nfeats)
        self.imgproj = nn.Linear(self.nimgfeats, self.nfeats)
        self.resproj = nn.Linear(nfeats * 2, self.nfeats)
        # self.emb_1 = nn.Linear(nfeats, nfeats, dropout=dropout)
        # self.imgproj = nn.Linear(self.nimgfeats, self.nfeats, dropout=dropout)
        # self.resproj = nn.Linear(nfeats * 2, self.nfeats, dropout=dropout)

        n_in = 2 * self.nfeats
        n_out = self.nfeats
        self.n_layers = num_layers
        self.convs = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.kernel_size = 5
        self.pad = self.kernel_size - 1
        for i in range(self.n_layers):
            # self.convs.append(Conv1d(n_in, 2 * n_out, self.kernel_size, self.pad, dropout))
            self.convs.append(nn.Conv1d(n_in, 2 * n_out, self.kernel_size, padding=self.pad))
            self.attention.append(AttentionLayer(n_out, nfeats))
            n_in = n_out

        self.classifier_0 = nn.Linear(self.nfeats, (nfeats // 2))
        # self.classifier_1 = nn.Linear((nfeats // 2), num_wordclass, dropout=dropout)
        self.classifier_1 = nn.Linear((nfeats // 2), num_wordclass)

    def forward(self, images, captions):
        imgsfeats = self.features_nopool(images)
        imgsfeats_pool = self.features_pool(imgsfeats)
        imgsfeats = imgsfeats_pool.view(imgsfeats_pool.size(0), -1)
        imgsfc7 = self.classifier(imgsfeats)

        attn_buffer = None
        wordemb = self.emb_0(captions)
        wordemb = self.emb_1(wordemb)
        x = wordemb.transpose(2, 1)
        batchsize, wordembdim, maxtokens = x.size()
        y = F.relu(self.imgproj(imgsfc7))
        y = y.unsqueeze(2).expand(batchsize, self.nfeats, maxtokens)
        x = torch.cat([x, y], 1)

        for i, conv in enumerate(self.convs):
            if i == 0:
                x = x.transpose(2, 1)
                residual = self.resproj(x)
                residual = residual.transpose(2, 1)
                x = x.transpose(2, 1)
            else:
                residual = x

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            x = x[:, :, :-self.pad]
            x = F.glu(x, dim=1)

            attn = self.attention[i]
            x = x.transpose(2, 1)
            x, attn_buffer = attn(x, wordemb, imgsfeats_pool)
            x = x.transpose(2, 1)

            x = (x + residual) * math.sqrt(.5)
        x = x.transpose(2, 1)
        x = self.classifier_0(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier_1(x)
        x = x.transpose(2, 1)

        return x, attn_buffer

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.vgg.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune