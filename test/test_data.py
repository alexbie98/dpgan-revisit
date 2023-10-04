import pytest

import torch

import dpgan.data

def test_get_mnist_reprod():
    train_dataset, val_dataset, img_dim, num_labels = dpgan.data.get_dataset(
        'MNIST', 'val', 0, torch.device('cuda:0'), True
    )

    assert img_dim == (1,28,28)
    assert num_labels == 10

    checksum =  [
        val_dataset[10][0].sum(),
        train_dataset[10][0].sum(),
        train_dataset[6123][0].mean() * train_dataset[1][1]
    ]
    checksum_expect = [118.30195617675781, 100.9686279296875, 0.7308824062347412]

    assert checksum == checksum_expect

def test_get_dataset_raises():
    with pytest.raises(KeyError):
        dpgan.data.get_dataset('ImageNet', 'val', 0, torch.device('cuda:0'), True)

    with pytest.raises(NotImplementedError):
        dpgan.data.get_dataset('MNIST', 'train', 1, torch.device('cuda:0'), False)
