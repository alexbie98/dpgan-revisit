import time
import tqdm

import torch

import dpgan.models
import dpgan.utils


def run_eval(g, num_gen_examples_eval, val_dataset, dataset, val_set, num_labels, seed, device):
    start_time = time.perf_counter()

    print(f'| Generating dataset of {num_gen_examples_eval} examples:')
    fake_img, fake_label = dpgan.eval.generate_dataset(
        g=g,
        num_gen_examples_eval=num_gen_examples_eval,
        gen_bsz=500,
        device=device
    )

    print('| Calculating FID:')
    fid = dpgan.eval.calculate_fid(
        fake_img,
        val_dataset.tensors[0],
        dataset=dataset,
        val_set=val_set,
        bsz=128,
        device=device
    )

    print(f'| FID: {fid}')

    print('| Calculating acc:')
    acc = dpgan.eval.calculate_acc(
        fake_img,
        fake_label,
        val_dataset.tensors[0],
        val_dataset.tensors[1],
        num_labels=num_labels,
        device=device,
        seed=seed
    )

    print(f'| total val time elapsed: {time.perf_counter() - start_time}')

    return fid, acc

def generate_dataset(g: dpgan.models.LabelledGenerator, num_gen_examples_eval, gen_bsz, device):

    # assert num_gen_examples_eval % gen_bsz == 0

    fake_img = []
    fake_label = []

    # sample a synthetic dataset
    g.eval()
    with torch.no_grad():
        for _ in tqdm.tqdm(range(num_gen_examples_eval//gen_bsz)):
            fake_img_batch, fake_label_batch = g.sample(gen_bsz, device)
            fake_img.append(fake_img_batch.detach().cpu())
            fake_label.append(fake_label_batch.detach().cpu())

        leftover = num_gen_examples_eval - ((num_gen_examples_eval//gen_bsz) * gen_bsz)
        final_fake_img_batch, final_fake_label_batch = g.sample(leftover, device)
        fake_img.append(final_fake_img_batch.detach().cpu())
        fake_label.append(final_fake_label_batch.detach().cpu())

    g.train()

    fake_img = dpgan.utils.round_normalized_img(torch.cat(fake_img))
    fake_label = torch.cat(fake_label)

    assert len(fake_img) == len(fake_label)
    assert len(fake_img) == num_gen_examples_eval

    return fake_img, fake_label
