import math
import numpy as np

import torch

import dpgan.utils

def test_count_split():
    assert dpgan.utils.count_split(100, 5) == [20, 20, 20, 20, 20]
    assert dpgan.utils.count_split(101, 5) == [21, 20, 20, 20, 20]

    assert dpgan.utils.count_split(136, 9) == [16, 15, 15, 15, 15, 15, 15, 15, 15]

    def test_match_np(total, max_size):
        data = np.ones(total)
        lens = [len(x) for x  in np.array_split(data, math.ceil(len(data)/max_size))]
        assert lens == dpgan.utils.count_split(total, math.ceil(len(data)/max_size))

    test_match_np(100, 10)
    test_match_np(5500, 35)
    test_match_np(136, 16)
    test_match_np(128, 16)


def gnorm_setup():

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = torch.nn.Linear(1,1)
            with torch.no_grad():
                self.lin1.weight[0][0] = 1.
                self.lin1.bias[0] = 1.

            self.lin1.bias.requires_grad = False
            self.lin2 = torch.nn.Linear(1,1, bias = False)

        def forward(self, x):
            return self.lin2(self.lin1(x))

    n = Net()
    x = torch.Tensor([[2],[1]])

    return n, x

def test_calc_gnorm():

    n, x = gnorm_setup()

    loss = n(x).mean()
    loss.backward()

    a = n.lin2.weight[0][0].item()
    g = np.array([(1+1) + (2+1), (a*1) + (a*2)])/2

    assert abs(dpgan.utils.calc_gnorm(n) - np.linalg.norm(g)) < 1e-5

def test_grad_vec():
    n, x = gnorm_setup()

    # accum 2 batches, then take gvec
    n.zero_grad()
    loss1 = n(x[0][None]).mean()
    loss1.backward()
    loss2 = n(x[1][None]).mean()
    loss2.backward()
    gvec = dpgan.utils.get_gvec(n)/2

    # 1 batch + gvec, 1 batch + gvec accumulated
    n.zero_grad()
    loss1 = n(x[0][None]).mean()
    loss1.backward()
    gvec1 = dpgan.utils.get_gvec(n)
    n.zero_grad()
    loss2 = n(x[1][None]).mean()
    loss2.backward()
    gvec2 = dpgan.utils.get_gvec(n)
    gvec_accum = (gvec1 + gvec2)/2

    # combined batch + gnorm
    n.zero_grad()
    loss = n(x).mean()
    loss.backward()
    gnorm = dpgan.utils.calc_gnorm(n)

    assert torch.isclose(gvec.norm(), gnorm)
    assert torch.isclose(gvec_accum.norm(), gnorm)

def avg_setup():

    torch.manual_seed(0)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = torch.nn.Linear(2, 2)
            self.lin2 = torch.nn.Linear(2, 1)

        def forward(self, x):
            return self.lin2(torch.relu(self.lin1(x)))

    n = Net()

    expected_weights = [
        torch.Tensor([[-0.0053,  0.3793], [-0.5820, -0.5204]]),
        torch.Tensor([-0.2723,  0.1896]),
        torch.Tensor([[-0.0140,  0.5607]]),
        torch.Tensor([-0.0628])
    ]

    # seed check
    for x, y in zip(expected_weights, n.parameters()):
        assert x.isclose(y, 1e-2).all()

    return n, expected_weights

def test_model_avg():

    n, expected_weights = avg_setup()

    # zeros_like test
    weights = dpgan.utils.weights_zeros_like(n)
    assert len(weights) == 4
    for w in weights:
        assert w.sum() == 0

    # adding scaled
    dpgan.utils.weights_add_scaled(weights, n, 0.1)
    for w, ew in zip(weights, expected_weights):
        assert w.isclose(0.1*ew, 1e-2).all()

    # model assignment
    dpgan.utils.weights_assign_model(weights, n)
    for p, ew in zip(n.parameters(), expected_weights):
        assert p.isclose(0.1*ew, 1e-2).all()

    # reset
    weights = dpgan.utils.weights_zeros_like(n)
    for w in weights:
        assert w.sum() == 0

    # check does not affect n parameters
    for p, ew in zip(n.parameters(), expected_weights):
        assert p.isclose(0.1*ew, 1e-2).all()

def test_model_ema():

    n, expected_weights = avg_setup()

    # copy test
    weights = dpgan.utils.weights_copy(n)
    assert len(weights) == 4
    for w, p in zip(weights, n.parameters()):
        assert (w == p.data).all()
        assert not w.requires_grad

    # simulate an update
    with torch.no_grad():
        for p in n.parameters():
            p.data = torch.ones_like(p.data)

    # verify copy no change
    for w, e_w in zip(weights, expected_weights):
        assert w.isclose(e_w, 1e-2).all()

    # ema update
    beta = 0.9
    dpgan.utils.weights_scale(weights, beta)
    dpgan.utils.weights_add_scaled(weights, n, 1-beta)

    # verify
    expected_ema = [beta*e_w + (1-beta) for e_w in expected_weights]
    for w, e_e in zip(weights, expected_ema):
        assert w.isclose(e_e, 1e-2).all()

    # verify model no change
    for p in n.parameters():
        assert (p.data == 1).all()


if __name__ == '__main__':
    test_model_ema()
