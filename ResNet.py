import torch, pdb, sys
from torch import nn
import torchvision
import torch.nn.functional as F


class ResNet(nn.Module):
    """
    Encoder.
    """

    def __init__(self):
        super(ResNet, self).__init__()
        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        modules = list(resnet.children())[0]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        feat = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        feat = torch.flatten(feat, 1)
        out = self.fc(feat)
        out_a = self.fc_a(feat)
        return out, out_a

    def fine_tune(self, fine_tune=True, finetune_start_layer=5):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = True
        # for p in self.resnet.parameters():
        #     p.requires_grad = False
        # # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        # for c in list(self.resnet.children())[finetune_start_layer:]:
        #     for p in c.parameters():
        #         p.requires_grad = fine_tune