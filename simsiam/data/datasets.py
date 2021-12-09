import torch
from torch.utils.data import Dataset


class DoubleAugmentDataset(Dataset):
    def __init__(self, dataset, transform, test_transform):
        self.dataset = dataset
        self.transform = transform
        self.test_transform = test_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x0 = self.test_transform(x)
        x1, x2 = self.transform(x), self.transform(x)
        return x0, torch.LongTensor([y]), x1, x2


class AugmentDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = self.transform(x)
        return x, torch.LongTensor([y])


class LinearDataset(Dataset):
    def __init__(self, f, z, y):
        self.f, self.z, self.y = f, z, y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        f, z = torch.FloatTensor(self.f[index]), torch.FloatTensor(self.f[index])
        y = torch.LongTensor([int(self.y[index])])
        return f, z, y