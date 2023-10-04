# dpgan-revisit

Techniques for improving the performance of differentially private (DP) GANs, as described in our paper:
- Alex Bie, Gautam Kamath*, Guojun Zhang*. [*Private GANs, revisited*](https://arxiv.org/abs/2302.02936). In TMLR, 2023. 

Tuning *n<sub>D</sub>* (number of D steps per G step) improves FID on MNIST.

<img src="https://github.com/alexbie98/dpgan-revisit/blob/main/figs/mnist-fid-vs-eps.png?raw=true" width="40%">

**Disclaimer**: this is *research code, not a production-grade DP implementation* suitable for releasing real sensitive data.
In particular, it does not handle issues like secure RNG and floating point vulnerabilities. 

## Requirements
- See `requirements.txt`
- Tested on: Python 3.11.4, PyTorch 2.0.1 (CUDA 11.7, cuDNN 8.5), **Opacus 1.1.3**
- **IMPORTANT**: tests fail with latest Opacus 1.4.0, seems like some semantics/breaking changes >1.1.3


## Quickstart

    $ pip install -e .              ## install
    $ python -m pytest test         ## run tests (optional)
    $ python train_dpgan.py         ## run ε=10 MNIST config
                                    ## requires ~15GB VRAM, runs in ~8 hours on 1x V100

See intermediate eval results and other diagnostics with TensorBoard. TensorBoard logs are saved in `logs/<dataset>/<run>/`. To view:

    $ tensorboard --logdir logs    ## then visit localhost:6006 in web browser

Checkpoints are saved in `results/<dataset>/<run>/`.

After training is done, to run FID and accuracy eval on a checkpoint:

    $ python scripts/eval_checkpoint.py --path results/<dataset>/<run>/<g-checkpoint>.pt

By default, this: (1) creates folders of `.png` files for real and generated data; (2) runs `pytorch-fid` and classifier training from the folders.
You can add the `--in_memory` flag to skip this step, which leads to similar but not identical numbers.


## Running different configs

Write your own training configurations in `config.yaml`. To use it, run

    $ python train_dpgan.py --config config.yaml

See [`exp_configs/example.yaml`](exp_configs/example.yaml) for an example config file that you can modify.

Some important configs you might want to experiment with:
- `bsz` (expected batch size)
- `num_d_steps` (total number of discriminator steps)
- `d_steps_per_g_step` (frequency of taking generator steps, relative to discriminator steps)
- `dp` (toggles between DP training and not)
- `sigma` (noise multiplier for DP)
- `max_physical_bsz` (used for simulate large batch sizes, experiment with this on your setup to maximize throughput without OOM)
- `ds` (enables adaptive discriminator step frequency)

Some settings used in the paper can be found in [`exp_configs/`](exp_configs/).


## Benchmarks
Selected benchmark numbers obtained by running configs in this repo.

|  ε | Dataset | Adaptive? | FID  | Acc. | Mem  | Config |
|:--:|---------|:---------:|-----:|-----:|-----:|:-----|
| ∞  | MNIST   |     ✘     |  3.4 | 97.1 |  6GB | [`mnist-nonpriv.yaml`](exp_configs/mnist-nonpriv.yaml) |
| | | | | | | |
| 10 | MNIST   |     ✘     | 19.4 | 93.0 | 15GB | [`mnist-eps10-50dsteps.yaml`](exp_configs/mnist-eps10-50dsteps.yaml) |
| 10 | MNIST   |     ✔     | ---- | ---- | 25GB | [`mnist-eps10-adaptive.yaml`](exp_configs/mnist-eps10-adaptive.yaml) |




## Acknowledgments

Repo structure from:
- Patrick J. Mineault & The Good Research Code Handbook Community. *The Good Research Code Handbook*. Zenodo. [doi:10.5281/zenodo.5796873](https://dx.doi.org/10.5281/zenodo.5796873). 2021.

Original non-private GAN implementation is adapted from Hyeonwoo Kang's code:
- [https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN](https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN).

which is an implementation of DCGAN:
- Alec Radford, Luke Metz, Soumith Chintala. [*Unsupervised representation learning with deep convolutional generative adversarial networks*](https://arxiv.org/abs/1511.06434). In ICLR 2016.

This implementation makes heavy use of [Opacus](https://github.com/pytorch/opacus):
- Ashkan Yousefpour, Igor Shilov, Alexandre Sablayrolles, Davide Testuggine, Karthik Prasad, Mani Malek, John Nguyen, Sayan Ghosh, Akash Bharadwaj, Jessica Zhao, Graham Cormode, Ilya Mironov. [*Opacus: User-friendly differential privacy library in PyTorch*](https://arxiv.org/abs/2109.12298). 2021.


## Citing

If you found this code useful, please consider citing us:

    @article{dpgan-revisit,
      title   = {Private {GAN}s, revisited},
      author  = {Alex Bie and
                 Gautam Kamath and
                 Guojun Zhang},
      journal = {Trans. Mach. Learn. Res.},
      volume  = {2023},
      year    = {2023},
      url     = {https://openreview.net/forum?id=9sVCIngrhP}
    }

