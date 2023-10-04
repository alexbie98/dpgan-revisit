import argparse
import time
import os
import tqdm
import numpy as np
import PIL.Image

import torch
import torchvision.transforms

import dpgan.exp
import dpgan.data
import dpgan.eval

def png_to_tensor(to_tensor, path):
    return to_tensor(PIL.Image.open(path))

def files_to_tensors(path, device):
    '''
    Given a directory with:
        - path/img/{i}.png
        - path/label.npy
    Produce (img, label) tensors with images scaled to [0...1]
    '''
    png_names = os.listdir(os.path.join(path, 'img'))
    png_names.sort()
    png_paths = [os.path.join(path, 'img', png_name) for png_name in png_names]

    to_tensor = torchvision.transforms.ToTensor()

    print(f'| loading {len(png_paths)} images @ {path}')

    img = list(tqdm.tqdm(png_to_tensor(to_tensor, p) for p in png_paths))

    img = torch.cat([x[None] for x in img]).to(device)
    label = torch.Tensor(np.load(os.path.join(path, 'label.npy'))).long().to(device)
    assert len(img) == len(label)
    return img, label


def get_datasets(path, name, val_set, seed, device):

    _, val_dataset, img_dim, num_labels = dpgan.data.get_dataset(
        name=name,
        val_set=val_set,
        seed=seed,
        device=device,
        in_memory=True
    )

    train_img, train_label = files_to_tensors(path, device)

    return train_img, train_label, val_dataset.tensors[0], val_dataset.tensors[1], img_dim, num_labels


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument(
        '--path', type=str,
        help='Point to a directory containing `img/{i}.png` and `labels.npy`',
        default=None
    )
    parser.add_argument('--dataset', type=str, default ='MNIST', choices=['MNIST', 'FashionMNIST', 'celeba'])
    parser.add_argument('--val_set', type=str, default='test', choices=['val', 'test'])
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    dpgan.exp.fix_randomness(args.seed)
    device = torch.device(f'cuda:{args.device}')

    print('| args:')
    print(args)

    if args.path is None:
        args.path = f'./data/{args.dataset}/train-{args.val_set}_gen/'

    train_img, train_label, val_img, val_label, _, num_labels = get_datasets(
        args.path, args.dataset, args.val_set, args.seed, device
    )

    start_time = time.perf_counter()
    acc = dpgan.eval.calculate_acc(
        train_img=train_img,
        train_label=train_label,
        val_img=val_img,
        val_label=val_label,
        num_labels=num_labels,
        device=device,
        seed=args.seed,
    )
    print(f'| time elapsed: {time.perf_counter() - start_time}')
    print(f'| acc: {acc}')


if __name__ == '__main__':
    main()
