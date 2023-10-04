import time
import os

import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms

DATASET_CONFIG = {
    'MNIST' : {
        'img_dim': (1,28,28),
        'num_labels': 10,
        'num_train': 60000,     # size of full train set
        'num_test': 10000,      # size of full test set
        'num_val': 5000,        # if we use val_set: val, take this amount from the train_set
                                # and use for validation (e.g. val = 5000, train = 60000 -> val=5000, train=55000)
        'labels': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        'transform': torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]),
        'checksum': '118.30195617675781, 100.9686279296875, 0.7308824062347412',
    },
    'FashionMNIST' : {
        'img_dim': (1,28,28),
        'num_labels': 10,
        'num_train': 60000,     # size of full train set
        'num_test': 10000,      # size of full test set
        'num_val': 5000,        # if we use val_set: val, take this amount from the train_set
                                # and use for validation (e.g. val = 5000, train = 60000 -> val=5000, train=55000)
        'labels': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        'transform': torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]),
        'checksum': '125.25882720947266, 384.20001220703125, 1.4991246461868286',
    },
    'celeba': {
        'img_dim': (3,32,32),
        'num_train': 182637,
        'num_test': 19962,
        'num_labels': 2,
        'labels': ['F', 'M'],
        'transform': torchvision.transforms.Compose([
            torchvision.transforms.Resize((32,32)),
            torchvision.transforms.ToTensor()
        ]),
        'checksum': '915.3490600585938, 1404.6705322265625, 0.0',
    }
}

def identity(x):
    return x

def dataset_to_tensor(dataset, device, path):

    if not os.path.exists(path):
        print('| converting...')
        t = time.perf_counter()
        x, y =  zip(*list(dataset))
        x = torch.stack(x)
        y = torch.Tensor(y).long()
        print(f'| time elapsed: {time.perf_counter() - t}')
        torch.save({'x': x, 'y': y}, path)
    else:
        print(f'| saved dataset tensor found at: {path}')
        saved = torch.load(path)
        x = saved['x']
        y = saved['y']

    return x.to(device), y.to(device)

def get_dataset(name, val_set, seed, device, in_memory=False, custom_transform=None):

    d_config = DATASET_CONFIG[name]
    transform = d_config['transform'] if custom_transform is None else custom_transform
    if name == 'MNIST':
        train_dataset = torchvision.datasets.MNIST('data', train=True, download=True,
                                                transform=transform)
        test_dataset = torchvision.datasets.MNIST('data', train=False, download=True,
                                                transform=transform)
    elif name == 'FashionMNIST':
        train_dataset = torchvision.datasets.FashionMNIST('data', train=True, download=True,
                                                transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST('data', train=False, download=True,
                                                transform=transform)
    elif name == 'celeba':
        # download=False, since link is broken
        # see https://github.com/pytorch/vision/issues/2262#issuecomment-1235752527
        # download all files manually to ./data/celeba, and unzip img_align_celeba.zip
        train_dataset = torchvision.datasets.CelebA('data', split='train', target_type = 'attr', download=False,
                                                transform=transform, target_transform=lambda y: y[20])
        val_dataset = torchvision.datasets.CelebA('data', split='valid', target_type = 'attr', download=False,
                                                transform=transform, target_transform=lambda y: y[20])

        train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        test_dataset = torchvision.datasets.CelebA('data', split='test', target_type = 'attr', download=False,
                                                transform=transform, target_transform=lambda y: y[20])
    else:
        raise NotImplementedError

    assert d_config['num_train'] == len(train_dataset)
    assert d_config['num_test'] == len(test_dataset)

    print(f'| loading dataset: {name}')

    if val_set ==  'val':
        print('| val_set: val')
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [d_config['num_train']-d_config['num_val'], d_config['num_val']],
            generator = torch.Generator().manual_seed(seed)
        )
    elif val_set == 'test':
        print('| val_set: test')
        val_dataset = test_dataset
    else:
        raise NotImplementedError

    print(f'| num_train: {len(train_dataset)}')
    print(f'| num_val:   {len(val_dataset)}')

    if in_memory:
        print('| storing dataset on GPU memory')
        train_dataset = torch.utils.data.TensorDataset(
            *dataset_to_tensor(train_dataset, device,
                               os.path.join('data', name,f'train_{val_set}.pt'))
        )
        val_dataset = torch.utils.data.TensorDataset(
            *dataset_to_tensor(val_dataset, device,
                               os.path.join('data', name,f'{val_set}.pt'))
        )

    if custom_transform is None:
        print(f"| checksum expect: {d_config['checksum']}")
        print(f'| checksum value : {val_dataset[10][0].sum()}, ' +
                    f'{train_dataset[10][0].sum()}, ' +
                    f'{train_dataset[6123][0].mean() * train_dataset[1][1]}')

    return train_dataset, val_dataset, d_config['img_dim'], d_config['num_labels']

if __name__ == "__main__":
    get_dataset("FashionMNIST", "val", 0, True, torch.device('cuda:0'))
