import argparse
import yaml
import tqdm

import torch.utils.data
import torch.nn
import torch.optim

import opacus
import opacus.accountants

import dpgan.exp
import dpgan.data

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
    def forward(self, x):
        return self.linear(x)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='exp_configs/debug.yaml')
    parser.add_argument('--q', type=float, default = None)
    parser.add_argument('--bsz', type=int, default = None)
    parser.add_argument('--n', type=int, default = None)

    parser.add_argument('--simulate', action='store_true')

    args = parser.parse_args()
    with open(args.config, encoding='UTF-8') as f:
        c = yaml.safe_load(f)
    dpgan.exp.validate(c)
    dpgan.exp.fix_randomness(c['seed'])

    if args.q is None or args.bsz is None or args.n is None or args.simulate:

        # dataset
        train_dataset, _, _, _ = dpgan.data.get_dataset(c['dataset'], c['val_set'], c['seed'], torch.device('cpu'))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c['bsz'])

        # privacy engine
        n = Net()
        opt = torch.optim.SGD(n.parameters(), lr=0.01)
        privacy_engine = opacus.PrivacyEngine(accountant='rdp')
        n, opt, train_loader = privacy_engine.make_private(
            module=n,
            optimizer=opt,
            data_loader=train_loader,
            noise_multiplier=c['sigma'],
            max_grad_norm=c['clip']
        )
        q = train_loader.sample_rate
        bsz = opt.expected_batch_size
        n = len(train_dataset)
    else:
        q = args.q
        bsz = args.bsz
        n = args.n

    # calc and log parameters
    print(
        f'| n = {n}, ' +
          f'q = {q}, '
          f'T = {c["num_d_steps"]} ' +
          f'sigma = {c["sigma"]}, ' +
          f'delta = {c["delta"]}, ' +
          f'expected bsz = {bsz} (correct = {q *n})'
    )

    if args.simulate:
        x = torch.zeros(1,1)
        for _ in tqdm.tqdm(range(1,c['num_d_steps']+1)):
            opt.zero_grad()
            n(x).squeeze().backward()
            opt.step()
        print(f'| eps = {privacy_engine.get_epsilon(c["delta"])}')

    else:
        rdp = opacus.accountants.rdp.privacy_analysis.compute_rdp(
            q=q,
            noise_multiplier=c['sigma'],
            steps=c['num_d_steps'],
            orders=opacus.accountants.RDPAccountant.DEFAULT_ALPHAS
        )
        eps = opacus.accountants.rdp.privacy_analysis.get_privacy_spent(
            orders=opacus.accountants.RDPAccountant.DEFAULT_ALPHAS,
            rdp=rdp,
            delta=c['delta']
        )[0]
        print(f'| eps = {eps}')


if __name__ == "__main__":
    main()

# celeba, 64:   --n 182637 --q 0.000350385423966363   --bsz 63
# celeba, 128:  --n 182637 --q 0.000700770847932726   --bsz 127
# celeba, 256:  --n 182637 --q 0.0014005602240896359  --bsz 255
# celeba, 512:  --n 182637 --q 0.0028011204481792717  --bsz 511
# celeba, 1024: --n 182637 --q 0.00558659217877095    --bsz 1020
# celeba, 2048: --n 182637 --q 0.011111111111111112   --bsz 2029
