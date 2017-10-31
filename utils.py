import pickle
import torch
from torch.utils import data
from torchvision import datasets, transforms


def get_dataset_mean_and_std(directory):
    dataset = datasets.ImageFolder(
        directory,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    data_loader = data.DataLoader(dataset)

    mean = [0, 0, 0]
    std = [0, 0, 0]
    for channel in range(3):
        _mean = 0
        _std = 0
        for _, (xs, _) in enumerate(data_loader):
            img = xs[0][channel].numpy()
            _mean += img.mean()
            _std += img.std()

        mean[channel] = _mean/len(dataset)
        std[channel] = _std/len(dataset)

    return mean, std


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        print('[INFO] Object saved to {}'.format(path))


def save_net(model, path):
    torch.save(model.state_dict(), path)
    print('[INFO] Checkpoint saved to {}'.format(path))


def load_net(model, path):
    model.load_state_dict(torch.load(path))
    print('[INFO] Checkpoint {} loaded'.format(path))
