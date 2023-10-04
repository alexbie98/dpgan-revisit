import contextlib
import argparse
import os
import yaml

import torch
import torch.utils.data
import torch.nn.functional as F

import opacus
import opacus.data_loader

# slighly modified version of opacus.utils.batch_memory_manager
import dpgan.data.batch_memory_manager

import dpgan.exp
import dpgan.data
import dpgan.loss
import dpgan.models
import dpgan.utils
import dpgan.metrics
import dpgan.eval
import dpgan.sched


def train_d_step(g: dpgan.models.Generator, d: dpgan.models.Discriminator,
           g_optimizer, d_optimizer, img, label, fake_bsz, device):

    g_optimizer.zero_grad()
    d_optimizer.zero_grad(set_to_none=True)
    g.zero_grad()
    d.zero_grad(set_to_none=True)

    fake_img, fake_label = [x.detach() for x in g.sample(fake_bsz, device)]

    scores = d(torch.cat([img, fake_img]), torch.cat([label, fake_label]))
    d_loss = F.binary_cross_entropy_with_logits(
        scores,
        torch.cat([torch.ones(len(img)), torch.zeros(len(fake_img))]).to(device),
        reduction='none'
    )
    d_loss.mean().backward()
    d_optimizer.step()

    g_optimizer.zero_grad()
    d_optimizer.zero_grad(set_to_none=True)
    g.zero_grad()
    d.zero_grad(set_to_none=True)

    return scores.detach().cpu(), d_loss.detach().cpu()


def val_d_step(g: dpgan.models.Generator, d: dpgan.models.Discriminator, img, label, device):

    g.zero_grad()
    d.zero_grad(set_to_none=True)
    g.eval()
    d.eval()

    with torch.no_grad():
        fake_img, fake_label = [x.detach() for x in g.sample(len(img), device)]

        scores = d(torch.cat([img, fake_img]), torch.cat([label, fake_label]))
        d_loss = F.binary_cross_entropy_with_logits(
            scores,
            torch.cat([torch.ones(len(img)), torch.zeros(len(fake_img))]).to(device),
            reduction = 'none'
        )

    d.train()
    g.train()
    g.zero_grad()
    d.zero_grad(set_to_none=True)

    return scores.detach().cpu(), d_loss.detach().cpu()


def train_g_step(g: dpgan.models.Generator, d: dpgan.models.Discriminator,
           g_optimizer, d_optimizer, g_loss_fn, bsz, device, is_private):

    g_optimizer.zero_grad()
    d_optimizer.zero_grad(set_to_none=True)
    g.zero_grad()
    d.zero_grad(set_to_none=True)

    if is_private:
        d.disable_hooks()

    fake_img, fake_label = g.sample(bsz, device)

    fake_scores = d(fake_img, fake_label)
    g_loss = g_loss_fn(fake_scores)
    g_loss.mean().backward()
    g_optimizer.step()

    if is_private:
        d.enable_hooks()

    g_optimizer.zero_grad()
    d_optimizer.zero_grad(set_to_none=True)
    g.zero_grad()
    d.zero_grad(set_to_none=True)

    return fake_scores.detach().cpu(), g_loss.detach().cpu()


def accumulate_train_d_metrics(old_m, scores, d_loss, real_bsz):

    bsz = len(d_loss)
    fake_bsz = bsz - real_bsz
    assert len(scores) == bsz

    # accumulate metrics
    new_m = dpgan.metrics.accumulate(
        old_m,
        {
            'd_loss': (d_loss.sum().item(), bsz),
            'd_loss/real': (d_loss[:real_bsz].sum().item(), real_bsz),
            'd_loss/fake': (d_loss[real_bsz:].sum().item(), fake_bsz),
            'd_acc': ((torch.sigmoid(scores[:real_bsz]) >= 0.5).sum().item() +
                      (torch.sigmoid(scores[real_bsz:]) < 0.5).sum().item(), bsz),
            'd_acc/real': ((torch.sigmoid(scores[:real_bsz]) >= 0.5).sum().item(), real_bsz),
            'd_acc/fake': ((torch.sigmoid(scores[real_bsz:]) < 0.5).sum().item(), fake_bsz),
            'bsz/real_mean_physical': (real_bsz, 1),
            'bsz/real': (real_bsz, 0),
            'bsz/fake': (fake_bsz, 0),
            'bsz': (bsz, 0),
        }
    )
    assert new_m['d_acc/real'][0] + new_m['d_acc/fake'][0] == new_m['d_acc'][0]
    assert new_m['d_acc/real'][1] + new_m['d_acc/fake'][1] == new_m['d_acc'][1]

    return new_m


def log_val_d_metrics(scores, d_loss, real_bsz, writer, d_step, g_step):

    val_m = {
        'd_loss': 0,
        'd_loss/real': 0,
        'd_loss/fake': 0,
        'd_loss/fake-real': 0,
        'd_acc': 0,
        'd_acc/real': 0,
        'd_acc/fake': 0,
        'd_acc/fake-real': 0
    }
    # loss metrics
    d_loss_real = d_loss[:real_bsz].mean().item()
    d_loss_fake = d_loss[real_bsz:].mean().item()
    val_m['d_loss'] = d_loss.mean().item()
    val_m['d_loss/real'] = d_loss_real
    val_m['d_loss/fake'] = d_loss_fake
    val_m['d_loss/fake-real'] = d_loss_fake - d_loss_real

    # acc metrics
    scores_real = scores[:real_bsz]
    scores_fake = scores[real_bsz:]
    d_acc_real = (torch.sigmoid(scores_real) >= 0.5).sum().item() / len(scores_real)
    d_acc_fake = (torch.sigmoid(scores_fake) < 0.5).sum().item() / len(scores_fake)
    d_acc = ((torch.sigmoid(scores[:real_bsz]) >= 0.5).sum() +
             (torch.sigmoid(scores[real_bsz:]) < 0.5).sum()).item() / len(scores)
    val_m['d_acc'] = d_acc
    val_m['d_acc/real'] = d_acc_real
    val_m['d_acc/fake'] = d_acc_fake
    val_m['d_acc/fake-real'] = d_acc_fake - d_acc_real

    for key, val in val_m.items():
        writer.add_scalar(key, val, d_step)
        writer.add_scalar('g_step_' + key, val, g_step)

    return d_acc, d_acc_real, d_acc_fake


def display_example_img(g, writer, d_step, c, device):

    g.eval()
    with torch.no_grad():
        fake_img, fake_label = g.sample(25, device)
    g.train()
    fake_img = dpgan.utils.round_normalized_img(fake_img)

    example_img_fig = dpgan.metrics.create_example_img(
        fake_img.detach().cpu().numpy(),
        fake_label.detach().cpu().numpy(),
        display_labels=dpgan.data.DATASET_CONFIG[c['dataset']]['labels']
    )
    writer.add_image('fake_img', example_img_fig, d_step)

# main -------------------------------------------------------------------------

def main():

    # experiment setup ---------------------------------------------------------
    parser = argparse.ArgumentParser()

    # main args in config file
    parser.add_argument(
        '--config', 
        type=str,
        default='exp_configs/mnist-eps10-50dsteps.yaml',
        help='See example config at exp_configs/example.yaml to set parameters'
    )

    # arguments for running checkpoint restarting experiment in the paper
    # not designed to be used for anything else
    # - use for saving and resuming training at your own (private users') risk
    parser.add_argument('--save_extra', action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--no_load_d', action='store_true')

    args = parser.parse_args()
    config = args.config
    with open(config, encoding='UTF-8') as f:
        c = yaml.safe_load(f)

    print(f'| using config located at path: {config}')

    # setup directories, tensorboard
    results_dir, train_writer, val_writer = dpgan.exp.exp_setup(c)

    # fix seed for reproducibility
    dpgan.exp.fix_randomness(c['seed'])

    # set gpu device
    device = torch.device(f'cuda:{c["gpu"]}')

    # data ---------------------------------------------------------------------
    train_dataset, val_dataset, img_dim, num_labels = dpgan.data.get_dataset(
        c['dataset'],
        c['val_set'],
        c['seed'],
        device,
        in_memory = True # in_memory keeps the data in gpu memory
    )

    # set the number of generated samples to evaluate on
    c['num_gen_examples_eval'] = (len(train_dataset) if c['num_gen_examples_eval'] is None
                                                     else c['num_gen_examples_eval'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=c['bsz'],
        shuffle=False, generator = torch.Generator().manual_seed(c['seed'])
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=c['bsz'],
        shuffle=True, generator = torch.Generator().manual_seed(c['seed'])
    )

    # model --------------------------------------------------------------------
    if img_dim == (1,28,28): # MNIST and FashionMNIST

        g = dpgan.models.DCLabelledGenerator(
            dim=c['dim_g'],
            img_dim=img_dim,
            dim_latent=c['dim_latent'],
            num_labels=num_labels,
            device=device
        ).to(device)

        d = dpgan.models.DCLabelledDiscriminator(
            dim=c['dim_d'],
            img_dim=img_dim,
            num_labels=num_labels,
            device=device
        ).to(device)

    elif img_dim == (3,32,32): # CelebA

        if c['small']:
            g = dpgan.models.DC32SmallLabelledGenerator(
                dim=c['dim_g'],
                img_dim=img_dim,
                dim_latent=c['dim_latent'],
                num_labels=num_labels,
                device=device
            ).to(device)

            d = dpgan.models.DC32SmallLabelledDiscriminator(
                dim=c['dim_d'],
                img_dim=img_dim,
                num_labels=num_labels,
                device=device
            ).to(device)
        else:
            g = dpgan.models.DC32LabelledGenerator(
                dim=c['dim_g'],
                img_dim=img_dim,
                dim_latent=c['dim_latent'],
                num_labels=num_labels,
                device=device
            ).to(device)

            d = dpgan.models.DC32LabelledDiscriminator(
                dim=c['dim_d'],
                img_dim=img_dim,
                num_labels=num_labels,
                device=device
            ).to(device)
    else:
        raise ValueError(f'unsupported \'img_dim\': {img_dim}')

    print('| Generator:')
    print(g)
    print(f'| G num_trainable_params: {dpgan.utils.num_params(g)}')

    print('| Discriminator:')
    print(d)
    print(f'| D num_trainable_params: {dpgan.utils.num_params(d)}')

    # optimizer ----------------------------------------------------------------
    g_optimizer = torch.optim.Adam(g.parameters(), lr=c['g_lr'], betas=(c['beta1'], c['beta2']))
    d_optimizer = torch.optim.Adam(d.parameters(), lr=c['d_lr'], betas=(c['beta1'], c['beta2']))

    # loading from checkpoint --------------------------------------------------

    if args.checkpoint: # used for checkpoint restarting experiments only
        print(f'| Loading checkpoint: {args.checkpoint}')
        state = torch.load(args.checkpoint)
        g_optimizer.load_state_dict(state['g_opt'])
        g.load_state_dict(state['g'])
        if args.no_load_d:
            print('| discriminator loaded from scratch')
        else:
            d_optimizer.load_state_dict(state['d_opt'])
            d.load_state_dict(state['d'])

    # adaptive discriminator frequency -----------------------------------------
    if c['ds']:
        print('| using discriminater step schedule:')
        print(
            f'| d_steps_per_g_step_init: {c["d_steps_per_g_step"]}, ' +
            f'grace: {c["grace"]}, ' +
            f'thresh: {c["thresh"]}, ' +
            f'ema β: {c["ds_beta"]}'
        )
        d_step_sched = dpgan.sched.DStepScheduler(
            d_steps_rate_init=c['d_steps_per_g_step'],
            grace=c['grace'],
            thresh=c['thresh'],
            beta=c['ds_beta']
        )
    else:
        print(f'| using constant d_steps_per_g_step: {c["d_steps_per_g_step"]}')
        d_step_sched = dpgan.sched.ConstDStepScheduler(
            d_steps_rate=c['d_steps_per_g_step'],
            beta=0.99
        )

    # privacy engine -----------------------------------------------------------
    if c['dp']:
        print(f'| using DP, sigma = {c["sigma"]}, C = {c["clip"]}, delta = {c["delta"]}')
        privacy_engine = opacus.PrivacyEngine(accountant='rdp', secure_mode=False)

        d, d_optimizer, poisson_train_loader = privacy_engine.make_private(
            module=d,
            optimizer=d_optimizer,
            data_loader=train_loader,
            noise_multiplier=c['sigma'],
            max_grad_norm=c['clip'],
            loss_reduction='mean',
            noise_generator=torch.Generator(device).manual_seed(c['seed'])
        )
        actual_bsz = round(poisson_train_loader.sample_rate * len(train_dataset))
        d_optimizer.expected_batch_size += actual_bsz # we will train with equal num of fake samples
    else:
        print('| not using DP')
        poisson_train_loader = opacus.data_loader.DPDataLoader.from_data_loader(train_loader)
        actual_bsz = round(poisson_train_loader.sample_rate * len(train_dataset))
    print(f'| using logical bsz: {actual_bsz} @ num_d_steps: {c["num_d_steps"]}')

    # physical bsz manager
    use_batch_memory_manager = False
    if c['dp']:
        if c['bsz'] < 0.8 * c['max_physical_bsz']:
            print(f'| not using batch memory manager, bsz({c["bsz"]})<0.8*max_physical_bsz({c["max_physical_bsz"]})')
            batch_manager_context = contextlib.nullcontext(poisson_train_loader)
        else:
            print(f'| using batch memory manager with max_physical_bsz: {c["max_physical_bsz"]}')
            batch_manager_context = dpgan.data.batch_memory_manager.BatchMemoryManager(
                data_loader=poisson_train_loader,
                max_physical_batch_size=c['max_physical_bsz'],
                optimizer=d_optimizer
            )
            use_batch_memory_manager = True
    else:
        batch_manager_context = contextlib.nullcontext(poisson_train_loader)

    # run the training loop
    with batch_manager_context as mem_poisson_train_loader:

        # physical batching upkeep
        last_physical_batch = False
        physical_batch_index = 0
        fake_physical_bsz_alloc = None

        # logging metrics
        m = dpgan.metrics.d_metrics_init()

        d_step = 0
        g_step = 0
        while d_step < c['num_d_steps']:

            for (img, label) in mem_poisson_train_loader:

                # upkeep for physical batching
                if use_batch_memory_manager:
                    last_physical_batch = not d_optimizer._step_skip_queue[-1] # pylint: disable=protected-access
                    assert (last_physical_batch ==
                            (physical_batch_index == mem_poisson_train_loader.batch_sampler.num_physical_batches-1))
                    if physical_batch_index == 0:
                        fake_physical_bsz_alloc = dpgan.utils.count_split(
                            actual_bsz,
                            mem_poisson_train_loader.batch_sampler.num_physical_batches
                        )
                    fake_physical_bsz = fake_physical_bsz_alloc[physical_batch_index]
                else:
                    last_physical_batch = True
                    fake_physical_bsz = actual_bsz

                # train discriminator on 1 physical batch
                scores, d_loss = train_d_step(
                    g, d,
                    g_optimizer, d_optimizer,
                    img, label, fake_physical_bsz,
                    device,
                )

                physical_batch_index += 1
                m = accumulate_train_d_metrics(m, scores, d_loss, len(img))

                # upon finishing a logical batch
                if last_physical_batch:
                    d_step += 1
                    d_step_sched.d_step()
                    physical_batch_index = 0

                    # sanity checks --------------------
                    logical_batch_size = (
                        mem_poisson_train_loader.batch_sampler.current_logical_batch_size if
                        use_batch_memory_manager else len(img)
                    )
                    assert logical_batch_size == m['bsz/real'][0]
                    assert actual_bsz == m['bsz'][0] - m['bsz/real'][0]

                    # log and reset metrics ------------
                    dpgan.metrics.publish(m, train_writer, d_step)
                    train_writer.add_scalar('epoch', d_step * actual_bsz/len(train_dataset), d_step) # epoch
                    m = dpgan.metrics.d_metrics_init()

                    # when we are going to take a g_step --------------------------
                    if d_step_sched.is_g_step_time():

                        # log val d_loss and d_acc before g_step -------------------
                        val_img, val_label = next(iter(val_loader))
                        assert len(val_img) == len(val_label) and len(val_img) == c['bsz']
                        val_scores, val_d_loss = val_d_step(g, d, val_img, val_label, device)
                        _, _, d_acc_fake, = log_val_d_metrics(
                            val_scores, val_d_loss, c['bsz'],
                            val_writer, d_step, g_step
                        )
                        # log step rate ------------------------
                        # after taking d_steps and g_steps (and before_step+1),
                        # log the step rate last used
                        train_writer.add_scalar('d_step_rate', d_step_sched.get_d_steps_rate(), g_step)
                        train_writer.add_scalar('d_step_rate/d_step', d_step_sched.get_d_steps_rate(), d_step)

                        # take g_step ---------------------------
                        _, g_loss = train_g_step(
                            g, d, g_optimizer, d_optimizer,
                            g_loss_fn=dpgan.loss.nonsaturating_logistic,
                            bsz=actual_bsz,
                            device=device,
                            is_private = c['dp']
                        )
                        g_step+=1
                        d_step_sched.g_step(d_acc=d_acc_fake)

                        # log d_acc ema used by scheduler
                        val_writer.add_scalar('g_step_d_acc/fake/ema', d_step_sched.ema, g_step-1)
                        train_writer.add_scalar('g_step', g_step, d_step)

                        # log metrics
                        g_loss = g_loss.mean().item()
                        train_writer.add_scalar('g_loss', g_loss, d_step)
                        train_writer.add_scalar('g_loss/g_step', g_loss, g_step-1)
                        if c['dp']:
                            eps = privacy_engine.get_epsilon(c['delta'])
                            train_writer.add_scalar('eps', eps, d_step)
                            train_writer.add_scalar('eps/g_step', eps, g_step)

                        val_writer.flush()
                        train_writer.flush()

                    # display
                    if d_step % c['display_interval'] == 0:
                        display_example_img(g, val_writer, d_step, c, device)

                    # validation
                    if d_step % c['val_interval'] == 0:
                        print('| Running eval')
                        fid, acc = dpgan.eval.run_eval(
                            g, c['num_gen_examples_eval'],
                            val_dataset, c['dataset'], c['val_set'], num_labels,
                            c['seed'], device
                        )
                        val_writer.add_scalar('fid', fid, d_step)
                        val_writer.add_scalar('fid/g_step', fid, g_step)

                        val_writer.add_scalar('acc', acc, d_step)
                        val_writer.add_scalar('acc/g_step', acc, g_step)

                        # save model
                        g_save_model_path = os.path.join(results_dir, f'g@dstep{d_step}.pt')
                        print(f'| saving model @ {g_save_model_path}')
                        torch.save(g.to(torch.device('cpu')), g_save_model_path)
                        g = g.to(device)

                        if args.save_extra: # used for the checkpoint restarting experiments
                            state_dict = {
                                'd': d.state_dict(),
                                'g': g.state_dict(),
                                'd_opt': d_optimizer.state_dict(),
                                'g_opt': g_optimizer.state_dict()
                            }
                            state_save_path = os.path.join(results_dir, f'state@dstep{d_step}.pt')
                            print(f'| saving state @ {state_save_path}')
                            torch.save(state_dict, state_save_path)

                    if d_step == c['num_d_steps']:
                        break

    val_writer.flush()
    train_writer.flush()

if __name__ == "__main__":
    main()
