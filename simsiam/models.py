import torchvision.models as models
from torch.nn import Sequential, Linear, Conv2d, BatchNorm1d, ReLU, MaxPool2d, Flatten, AdaptiveAvgPool2d
from torch import nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_resnet(num_classes, zero_init_residual=True):
    resnet = models.resnet18(num_classes=num_classes, zero_init_residual=zero_init_residual)
    resnet.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = Identity()
    return resnet


def get_encoder(n_p_layers: int = 3, emb_dim: int = 2048, out_bn: bool = True):
    resnet = get_resnet(num_classes=emb_dim)
    hid_dim = resnet.fc.weight.shape[1]

    # projector
    layers = list()
    for _ in range(n_p_layers - 1):
        layers.append(Linear(hid_dim, hid_dim, bias=False))
        layers.append(BatchNorm1d(hid_dim))
        layers.append(ReLU(inplace=True))
    layers.append(resnet.fc)
    if out_bn:
        layers[-1].bias.requires_grad = False
        layers.append(BatchNorm1d(emb_dim, affine=False))
    projector = Sequential(*layers)

    resnet.fc = Identity()
    return resnet, projector


def get_predictor(n_layers: int = 2, emb_dim: int = 2048, hid_dim: int = 512, out_bn: bool = True):
    layers = list()
    for _ in range(n_layers - 1):
        layers.append(Linear(emb_dim, hid_dim, bias=False))
        layers.append(BatchNorm1d(hid_dim))
        layers.append(ReLU(inplace=True))
    if not out_bn:
        layers.append(Linear(hid_dim, emb_dim))
    else:
        layers.append(Linear(hid_dim, emb_dim, bias=False))
        layers.append(BatchNorm1d(emb_dim))
    return Sequential(*layers)

