import argparse
import os
import shutil
import tqdm
import numpy as np

import torch
import torchvision

import dpgan.exp
import dpgan.data
import dpgan.eval
import dpgan.utils


# This script takes as input a dataset AND either:
#     - a generator checkpoint 'path/to/{checkpoint}.pt'; OR
#     - None
# Produces synthetic dataset at 'path/to/{checkpoint}_gen/img/{i}.png'
#                               'path/to/{checkpoint}_gen/label.npy'

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default = None)
    parser.add_argument('--dataset', type = str, choices =['MNIST', 'FashionMNIST', 'celeba'], default='MNIST')
    parser.add_argument('--val_set', type=str, choices=['val', 'test'], default='test')
    parser.add_argument('--num_examples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()
    return args

def save(digits, path, x):
    i, example = x
    img, label = example
    img.save(path + 'img/' + f'{i}'.zfill(digits) + '.png')
    return label

def dataset_to_files(dataset, path):

    shutil.rmtree(path, ignore_errors=True)

    digits = len(str(len(dataset)))
    os.makedirs(path+'img', exist_ok=True)

    labels = list(tqdm.tqdm(save(digits, path, x) for x in enumerate(dataset)))

    np.save(path + 'label.npy', np.array(labels))


def main():
    args = parse_args()

    dpgan.exp.fix_randomness(args.seed)
    device = torch.device(f'cuda:{args.device}')

    print(f'args: {args}')
    d_config = dpgan.data.DATASET_CONFIG[args.dataset]

    num_examples = args.num_examples
    if num_examples is None:
        num_examples = (d_config['num_train'] if args.val_set == 'test'
                                              else d_config['num_train'] - d_config['num_val'])

    print(f'| num examples: {num_examples}')

    match args.path:
        case None:
            print(f'| producing .pngs for {args.dataset}')

            train_path = f'data/{args.dataset}/train-{args.val_set}_gen/'
            val_path = f'data/{args.dataset}/{args.val_set}_gen/'

            if args.overwrite or (not os.path.exists(train_path)) or (not os.path.exists(val_path)):
            
                custom_transform = torchvision.transforms.Compose([])
                if args.dataset == 'celeba':
                    custom_transform = torchvision.transforms.Resize((32,32))

                train_dataset, val_dataset, _, _ = dpgan.data.get_dataset(
                    args.dataset,
                    args.val_set,
                    args.seed,
                    args.device,
                    in_memory=False,
                    custom_transform=custom_transform
                )
                if not os.path.exists(val_path):
                    dataset_to_files(val_dataset, val_path)

                if not os.path.exists(train_path):
                    dataset_to_files(train_dataset, train_path)
            else:
                print('| already exists, add --overwrite flag to regen')

        case checkpoint_path:
            assert checkpoint_path[-3:] == '.pt'
            train_path = checkpoint_path[:-3] + '_gen/'
            print(f'| producing .pngs for generator checkpoint at {train_path}')

            if args.overwrite or not os.path.exists(train_path):
                g = torch.load(checkpoint_path, map_location='cpu').to(device)
                g.set_device(device)
                fake_img, fake_label = dpgan.eval.generate_dataset(g, num_examples, 500, device)
                train_dataset = dpgan.data.TensorDataset(
                    fake_img,
                    fake_label,
                    transform = torchvision.transforms.ToPILImage()
                )

                dataset_to_files(train_dataset, train_path)
            else:
                print('| already exists, add --overwrite flag to regen')

if __name__ == '__main__':
    main()
