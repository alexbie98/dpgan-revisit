import os
import json
import shutil
import random

import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch

from dpgan.data import DATASET_CONFIG

LOGGED_PARAMS = [
    'seed',
    'g_lr', 'd_lr',
    'bsz', 'num_d_steps', 'd_steps_per_g_step',
    # 'dim_latent', 'dim_g'
    'dim_d', 'dp', 'sigma', # 'clip',
    'ds', 'grace', 'thresh', 'ds_beta', 'small'
]

def fix_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)

    np.random.seed(seed)
    random.seed(seed)

def exp_setup(config):
    '''Read config file and:
        - set up logs and results directories
        - configure logger
        - save exp config
        - set up tensorboard writers
    '''

    validate(config)

    exp_name = to_exp_name(config) + ('-test' if config['val_set'] == "test" else '-val')

    # create directories
    results_dir = os.path.join('results', config['dataset'], exp_name)
    tb_dir = os.path.join('logs', config['dataset'], exp_name)
    shutil.rmtree(results_dir, ignore_errors=True)
    os.makedirs(results_dir)
    shutil.rmtree(tb_dir, ignore_errors=True)
    os.makedirs(tb_dir)

    # log exp config
    print('| config:')
    print(json.dumps(config, indent=4))

    # save exp config
    config_path = os.path.join(results_dir, 'config.json')
    with open(config_path, 'w', encoding='UTF-8') as f:
        json.dump(config, f, indent=4)

    # setup tensorboard writers
    train_writer = SummaryWriter(os.path.join(tb_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(tb_dir, 'val'))

    # t=0 tb logging
    val_writer.add_text('config', json.dumps(config), 0)
    val_writer.flush()

    return results_dir, train_writer, val_writer


def to_exp_name(config):
    return "-".join([x[0]+str(x[1]) for x in config.items() if x[0] in LOGGED_PARAMS])


def validate(config):
    '''Perform sanity checks on the input configuration file
    '''

    # perf settings
    assert isinstance(config['gpu'], int)
    assert isinstance(config['max_physical_bsz'], int)
    assert isinstance(config['num_workers'], int)

    # exp settings
    assert isinstance(config['dataset'], str) and config['dataset'] in ['MNIST', 'FashionMNIST', 'celeba']
    assert isinstance(config['val_set'], str) and config['val_set'] in ['val', 'test']

    # eval
    assert isinstance(config['val_interval'], int)
    assert isinstance(config['display_interval'], int)
    assert type(config['num_gen_examples_eval']) in [type(None), int]

    # seed
    assert isinstance(config['seed'], int)

    # optimizer
    assert isinstance(config['g_lr'], float)
    assert isinstance(config['d_lr'], float)
    assert isinstance(config['beta1'], float)
    assert isinstance(config['beta2'], float)

    # training
    assert isinstance(config['bsz'], int)
    assert isinstance(config['num_d_steps'], int)
    assert isinstance(config['d_steps_per_g_step'], int)

    num_train = (DATASET_CONFIG[config['dataset']]['num_train'] if config['val_set'] == 'test'
                 else DATASET_CONFIG[config['dataset']]['num_train'] - DATASET_CONFIG[config['dataset']]['num_val'])
    assert config['bsz'] < num_train

    assert config['d_steps_per_g_step'] <= config['num_d_steps']
    assert config['num_d_steps'] % config['d_steps_per_g_step'] == 0

    assert config['val_interval'] <= config['num_d_steps']
    assert config['num_d_steps'] % config['val_interval'] == 0

    assert config['display_interval'] <= config['num_d_steps']
    assert config['num_d_steps'] % config['display_interval'] == 0

    # architecture
    assert isinstance(config['dim_latent'], int)
    assert isinstance(config['dim_g'], int)
    assert isinstance(config['dim_d'], int)
    assert isinstance(config['small'], bool)

    assert config['dim_d']%2 == 0

    # privacy
    assert isinstance(config['dp'], bool)
    if config['dp']:
        assert isinstance(config['sigma'], float)
        assert isinstance(config['clip'], float)
        assert isinstance(config['delta'], float)

        assert config['delta'] < 1/num_train

    # d step schedule
    assert isinstance(config['ds'], bool)
    if config['ds']:
        assert isinstance(config['grace'], int)

        assert isinstance(config['thresh'], float)
        assert config['thresh'] >= 0 and config['thresh'] <=1

        assert isinstance(config['ds_beta'], float)
        assert config['ds_beta'] >= 0 and config['ds_beta'] <=1

