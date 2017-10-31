import numpy as np
from torch.utils import data
from torchvision import datasets, transforms

import utils


# Spilt a dataset into training and testing dataset.
# Returns: train_loader, test_loader
def get_train_test_loader(directory, batch_size, testing_size=0.1, img_size=None):
    mean, std = utils.get_dataset_mean_and_std(directory)
    # print(mean, std)
    transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    if img_size is not None:
        transform.insert(0, transforms.Scale(img_size))

    dataset = datasets.ImageFolder(
        directory,
        transform=transforms.Compose(transform)
    )

    num_data = len(dataset)
    indices = list(range(num_data))
    split = int(np.floor(testing_size * num_data))

    # Shuffle
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = data.sampler.SubsetRandomSampler(train_idx)
    test_sampler = data.sampler.SubsetRandomSampler(test_idx)

    train_loader = data.DataLoader(
        dataset,
        num_workers=2,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True
    )

    test_loader = data.DataLoader(
        dataset,
        num_workers=2,
        batch_size=batch_size,
        sampler=test_sampler,
        drop_last=True
    )

    return train_loader, test_loader


# For Office31 datasets data_loader
def get_office31_dataloader(case, batch_size):
    print('[INFO] Loading datasets: {}'.format(case))
    datas = {
        'amazon': 'dataset/office31/amazon/images/',
        'dslr': 'dataset/office31/dslr/images/',
        'webcam': 'dataset/office31/webcam/images/'
    }
    means = {
        'amazon': [0.79235075407833078, 0.78620633471295642, 0.78417965306916637],
        'webcam': [0.61197983011509638, 0.61876474000372972, 0.61729662103473015],
        'dslr': [],
        'imagenet': [0.485, 0.456, 0.406]
    }
    stds = {
        'amazon': [0.27691643643313618, 0.28152348841965347, 0.28287296762830788],
        'webcam': [0.22763857108616978, 0.23339382150450594, 0.23722725519031848],
        'dslr': [],
        'imagenet': [0.229, 0.224, 0.225]
    }

    img_size = (227, 227)

    transform = [
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(means['imagenet'], stds['imagenet']),
    ]

    data_loader = data.DataLoader(
        dataset=datasets.ImageFolder(
            datas[case],
            transform=transforms.Compose(transform)
        ),
        num_workers=2,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return data_loader
