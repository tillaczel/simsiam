from torchvision import transforms


NORMALIZATION_PARAMS = {
    'CIFAR10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]}
}


def get_unsupervised_train_transform(normalize, normalize_bool):
    transform = [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ]
    if normalize_bool:
        transform.append(normalize)
    transform = transforms.Compose(transform)
    return transform


def get_supervised_train_transform(normalize, normalize_bool):
    transform = [
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    if normalize_bool:
        transform.append(normalize)
    transform = transforms.Compose(transform)
    return transform


def get_test_transform(normalize, normalize_bool):
    if normalize_bool:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        return transform
    return transforms.ToTensor()


def get_unsupervised_transforms(dset_name, normalize_bool):
    norm_params = NORMALIZATION_PARAMS[dset_name]
    normalize = transforms.Normalize(mean=norm_params['mean'],
                                     std=norm_params['std'])
    train_transform = get_unsupervised_train_transform(normalize, normalize_bool)
    test_transform = get_test_transform(normalize, normalize_bool)
    return train_transform, test_transform


def get_supervised_transforms(dset_name, normalize_bool):
    norm_params = NORMALIZATION_PARAMS[dset_name]
    normalize = transforms.Normalize(mean=norm_params['mean'],
                                     std=norm_params['std'])
    train_transform = get_supervised_train_transform(normalize, normalize_bool)
    test_transform = get_test_transform(normalize, normalize_bool)
    return train_transform, test_transform

