import argparse
import os
import gc

import torch

import dpgan.exp
import dpgan.eval
import dpgan.data

def to_train_dataset_files_path(path, dataset, val_set):
    if path is None:
        return f'data/{dataset}/train-{val_set}_gen/'

    assert path[-3:] == '.pt'
    return path[:-3]+'_gen/'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--in_memory', action='store_true')
    parser.add_argument('--dataset', choices=['MNIST', 'FashionMNIST', 'celeba'], default='MNIST')
    parser.add_argument('--num_examples', default = None)
    parser.add_argument('--val_set', choices=['val', 'test'], default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    # boilerplate
    args = parse_args()
    dpgan.exp.fix_randomness(args.seed)
    print(f'| args: {args}')

    if not args.in_memory:

        # generate train_data files
        generate_train_dataset_files_command = (
            'python scripts/gen_dataset_files.py ' +
            (f'--path {args.path} ' if args.path is not None else '') +
            (f'--num_examples {args.num_examples} ' if args.num_examples is not None else '') +
            f'--dataset {args.dataset} ' +
            f'--seed {args.seed} ' +
            f'--device {args.device}'
        )
        print(generate_train_dataset_files_command)
        os.system(generate_train_dataset_files_command)
        train_dataset_files_path = to_train_dataset_files_path(args.path, args.dataset, args.val_set)

        # generate test_dataset files
        generate_test_dataset_files_command = (
            'python scripts/gen_dataset_files.py ' +
            f'--dataset {args.dataset} ' +
            f'--seed {args.seed} '
            f'--device {args.device}'
        )
        os.system(generate_test_dataset_files_command)

        # fid comp
        eval_fid_command = (
            f'python -m pytorch_fid {train_dataset_files_path}img data/{args.dataset}/{args.val_set}_gen/img'
        )
        os.system(eval_fid_command)

        # acc comp
        eval_classifier_acc_command = (
            'python scripts/eval_classifier_acc_files.py ' +
            f'--path {train_dataset_files_path} ' +
            f'--dataset {args.dataset} '
            f'--seed {args.seed} ' +
            f'--device {args.device}'
        )
        os.system(eval_classifier_acc_command)

    else:
        # device
        device = torch.device(f'cuda:{args.device}')

        # load checkpoint
        g = torch.load(args.path).to(device)
        g.set_device(device)

        # num_examples
        d_config = dpgan.data.DATASET_CONFIG[args.dataset]
        num_gen_examples_eval = (d_config['num_train'] if args.val_set == 'test'
                                                       else d_config['num_train'] - d_config['num_val'])
        if args.num_examples is not None:
            num_gen_examples_eval = args.num_examples

        # load data
        _, val_dataset, _, num_labels = dpgan.data.get_dataset(
            args.dataset, args.val_set, args.seed, device, in_memory = True
        )

        del _
        gc.collect()
        torch.cuda.empty_cache()

        fid, acc = dpgan.eval.run_eval(
            g, num_gen_examples_eval,
            val_dataset, args.dataset, args.val_set, num_labels, 
            args.seed, device
        )

        print(f'| acc: {acc}, fid: {fid} for {args.path}')


if __name__ == '__main__':
    main()
