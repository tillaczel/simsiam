from torchvision import transforms
from PIL import ImageFilter
import random


NORMALIZATION_PARAMS = {
    'CIFAR10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]}
}


def get_train_transform(normalize):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ])
    return transform


def get_test_transform(normalize):
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return transform


def get_transforms(dset_name: str):
    norm_params = NORMALIZATION_PARAMS[dset_name]
    normalize = transforms.Normalize(mean=norm_params['mean'],
                                     std=norm_params['std'])
    train_transform = get_train_transform(normalize)
    test_transform = get_test_transform(normalize)
    return train_transform, test_transform

