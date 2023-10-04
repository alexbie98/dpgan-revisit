import os
import tqdm
import numpy as np
import torch.utils.data

from . import pytorch_fid


def get_activations(img, model, bsz, device):
    model.eval()

    dataset = torch.utils.data.TensorDataset(img, torch.zeros(len(img)))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=False)

    pred_arr = np.empty((len(img), 2048))

    start_idx = 0
    for batch, _ in tqdm.tqdm(data_loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_fid(fake_img, real_img, dataset, val_set, bsz, device):

    block_idx = pytorch_fid.inception.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = pytorch_fid.inception.InceptionV3([block_idx]).to(device)

    if fake_img.shape[1] == 1:
        fake_img = fake_img.repeat(1,3,1,1)
        real_img = real_img.repeat(1,3,1,1)

    act_fake = get_activations(fake_img, model, bsz, device)
    m_fake = np.mean(act_fake, axis=0)
    s_fake = np.cov(act_fake, rowvar=False)

    precomp_act_stats_path = f'./data/{dataset}/{val_set}_act_stats.npz'
    if os.path.isfile(precomp_act_stats_path):
        act_stats_real = np.load(precomp_act_stats_path)
        m_real = act_stats_real['m_real']
        s_real = act_stats_real['s_real']
    else:
        act_real = get_activations(real_img, model, bsz, device)
        m_real = np.mean(act_real, axis=0)
        s_real = np.cov(act_real, rowvar=False)
        np.savez(precomp_act_stats_path, m_real=m_real, s_real=s_real)

    fid_value = pytorch_fid.fid_score.calculate_frechet_distance(m_fake, s_fake, m_real, s_real)

    return fid_value
