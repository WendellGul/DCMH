import torch
from torch import nn
from torchvision import models
from models.basic_module import BasicModule


MODEL_DIR = 'data/vgg_net.mat'  # not use

# We use vgg19_bn pre-trained model as the image feature extractor
vgg_net = models.vgg19_bn(pretrained=True)


class ImgModule(BasicModule):
    def __init__(self, bit):
        super(ImgModule, self).__init__()
        self.module_name = "image_model"
        self.features = vgg_net.features
        self.classifier = nn.Sequential(
            *list(vgg_net.classifier)[:-1],
            nn.Linear(4096, bit)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape((-1, 25088))
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    print(list(vgg_net.classifier))
