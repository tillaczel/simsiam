from torchvision import transforms
from PIL import ImageFilter
import random


def get_train_transform(normalize, crop_size):
    # Copied from https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py
    # TODO: augmentations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
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


def get_transforms(crop_size: int):
    # TODO: normalization
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    train_transform = get_train_transform(normalize, crop_size)
    test_transform = get_test_transform(normalize)
    return train_transform, test_transform


# Copied from https://github.com/facebookresearch/simsiam/blob/main/simsiam/loader.py
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
