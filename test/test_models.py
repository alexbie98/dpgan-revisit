import torch

import dpgan.models
import dpgan.utils

def test_dc32shapes():

    img_dim = (3,32,32)
    device = torch.device('cpu')

    d = dpgan.models.DC32LabelledDiscriminator(128, img_dim, 2, device).to(device)
    g = dpgan.models.DC32LabelledGenerator(128, img_dim, 128, 2, device).to(device)

    x, y = g.sample(4, device)
    assert x.shape == (4,3,32,32)
    assert x.min() >= 0
    assert y.max() <= 1
    assert y.shape == (4,)
    assert d(x,y).shape == (4,)

def test_dc32smallshapes():

    img_dim = (3,32,32)
    device = torch.device('cpu')

    d = dpgan.models.DC32SmallLabelledDiscriminator(128, img_dim, 2, device).to(device)
    g = dpgan.models.DC32SmallLabelledGenerator(128, img_dim, 128, 2, device).to(device)

    print(dpgan.utils.num_params(d))
    print(dpgan.utils.num_params(g))

    x, y = g.sample(9, device)
    assert x.shape == (9,3,32,32)
    assert x.min() >= 0
    assert y.max() <= 1
    assert y.shape == (9,)
    assert d(x,y).shape == (9,)

if __name__ =='__main__':
    test_dc32smallshapes()
