import torch
import torch.utils.data
import torch.nn
import torch.nn.functional as F

import opacus

import dpgan.data.batch_memory_manager
import dpgan.utils



class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, bias=False)
        self.linear.weight = torch.nn.parameter.Parameter(
            torch.Tensor([[1,1]]),
            requires_grad=True
        )

    def forward(self, x):
        return self.linear(x)

def opacus_setup(sigma, clip):
    ''' Example setup for various tests
    '''

    device = torch.device('cuda:0')

    # real data -> real dataset -> real loader
    real_x = torch.Tensor(
        [[10,-30],
         [8,-29],
         [10, -27],
         [6.5, -27],
         [12, -30],
         [10, -30],
         [9, -30],
         [9, -32],
         [10, -29]]
    ).float().to(device)

    real_dataset = torch.utils.data.TensorDataset(real_x)

    real_loader = torch.utils.data.DataLoader(
        real_dataset,
        batch_size = 3,
        shuffle = True,
        generator = torch.Generator().manual_seed(0)
    )

    # reprod
    for x, in real_loader:
        expect = torch.Tensor(
            [[10, -27],
             [8, -29],
             [6.5, -27]]
        ).float().to(device)
        assert (expect == x).all()
        break

    # net
    net = Net().to(device)
    assert (net(expect).squeeze() == torch.Tensor([-17, -21, -20.5]).float().to(device)).all()

    # opt
    opt = torch.optim.SGD(net.parameters(), lr = 1.0)

    # dp
    privacy_engine = opacus.PrivacyEngine()
    dp_net, dp_opt, poisson_real_loader = privacy_engine.make_private(
        module=net,
        optimizer=opt,
        data_loader=real_loader,
        noise_multiplier=sigma,
        max_grad_norm=clip,
        loss_reduction='mean',
        noise_generator=torch.Generator(device).manual_seed(0)
    )

    return device, net, opt, real_x, real_loader, privacy_engine, dp_net, dp_opt, poisson_real_loader



def test_opacus_backward():
    ''' A synthetic example of what happens when opacus takes a backward step
    with mixed private and public data.
    '''

    device, _, _, _, _, privacy_engine, dp_net, dp_opt, poisson_real_loader = opacus_setup(1.0, 1.0)

    # fake data
    fake_x = torch.Tensor(
        [[10, 10],
         [9, 7],
         [11, 7]],
    ).float().to(device)

    x = None
    for i,(x,) in enumerate(poisson_real_loader):
        if i==0:
            continue
        dp_opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            dp_net(torch.cat([x, fake_x])).squeeze(),
            torch.cat([torch.ones(len(x)), torch.zeros(len(fake_x))]).to(device),
            reduction = 'none'
        )
        break

    # reprod sample
    expect = torch.Tensor(
        [[9, -30],
            [9, -32]]
    ).float().to(device)
    assert (x == expect).all()

    # for bce loss at large values, roughly approximated by:
    #         l(x) = -x if y = 1
    #                 x if y = 0
    expect = torch.cat([-torch.sum(x, axis=1), torch.sum(fake_x, axis=1)])
    assert (loss == expect).all()
    print(loss)

    # no grad yet
    assert dp_net.linear.weight.grad is None
    assert dp_net.linear.weight.grad_sample is None

    loss.mean().backward() # backward step

    # based on defn of l above, per-example gradient with respect to w can be calculated as
    expect = torch.cat([-x, fake_x]).unsqueeze(1) # B x d -> B x 1 x d
    assert dp_net.linear.weight.grad_sample.isclose(expect).all()
    print(dp_net.linear.weight.grad_sample)

    # normal grad is average of per-example grad
    assert dp_net.linear.weight.grad.isclose(expect.mean(axis=0)).all()
    print(dp_net.linear.weight.grad)

    dp_opt.step()

    # normalize + sum
    expect = F.normalize(expect, dim=2).squeeze().sum(axis=0)
    print(expect)
    # add noise + divide by expected batch size (3)
    expect = (
        expect +
        torch.normal(mean=0, std=1.0, size=(1,2), device=device, generator=torch.Generator(device).manual_seed(0))
    )/3

    assert dp_net.linear.weight.grad.isclose(expect).all()
    print(expect)
    print(dp_net.linear.weight.grad)

    # SGD step
    expect = torch.Tensor([[1,1]]).float().to(device) - 1.0 * expect
    assert dp_net.linear.weight.isclose(expect).all()
    print(expect)
    print(dp_net.linear.weight)

    # epsilon
    assert privacy_engine.get_epsilon(delta = 1e-5) == 3.413505963163778

def test_opacus_weight_tying():
    ''' A test to check opacus' behaviour when optimizing the original version of a dp_net.
        Conclusion: weights are shared
                    Forward pass with net will now always result in per-sample-grads being computed.
    '''
    device, net, _, real_x, _, _, dp_net, dp_opt, _ = opacus_setup(1.0, 1.0)

    # using dp_opt affects net
    assert (net.linear.weight.cpu().detach() == torch.Tensor([[1,1]])).all()

    loss = F.binary_cross_entropy_with_logits(
        dp_net(real_x).squeeze(),
        torch.ones(len(real_x)).to(device),
        reduction = 'none'
    )
    loss.mean().backward()
    dp_opt.step()
    dp_opt.zero_grad()

    expect = torch.Tensor([[2.2199633121, -1.7134380341]])
    assert  net.linear.weight.cpu().detach().isclose(expect).all()

    # net is converted into a grad sample module
    with torch.no_grad():
        net.linear.weight[0][0] = 1.
        net.linear.weight[0][1] = 1.

    assert (net.linear.weight.cpu().detach() == torch.Tensor([[1,1]])).all()
    assert net.linear.weight.grad_sample is None

    loss = F.binary_cross_entropy_with_logits(
        net(real_x).squeeze(),
        torch.ones(len(real_x)).to(device),
        reduction = 'none'
    )
    loss.mean().backward()

    assert net.linear.weight.grad_sample.shape == (len(real_x), 1, 2)

def test_opacus_enable_hooks():
    ''' A test to check opacus' behaviour when a dp_net, with hooks enabled, disabled.
        Conclusion: - hooks disabled: no grad sample
                    - hooks enabled: grad sample is computed
    '''
    device, _, _, real_x, _, _, dp_net, dp_opt, _ = opacus_setup(1.0, 1.0)

    # net is converted into a grad sample module
    with torch.no_grad():
        dp_net.linear.weight[0][0] = 1.
        dp_net.linear.weight[0][1] = 1.

    dp_net.disable_hooks()

    loss = F.binary_cross_entropy_with_logits(
        dp_net(real_x).squeeze(),
        torch.ones(len(real_x)).to(device),
        reduction = 'none'
    )
    loss.mean().backward()

    assert dp_net.linear.weight.grad_sample is None
    assert dp_net.linear.weight.grad[0].isclose(-real_x.mean(dim=0)).all()

    dp_net.zero_grad(set_to_none=True)

    dp_net.enable_hooks()
    loss = F.binary_cross_entropy_with_logits(
        dp_net(real_x).squeeze(),
        torch.ones(len(real_x)).to(device),
        reduction = 'none'
    )
    loss.mean().backward()

    expect = -real_x[:, None, :] # expected per sample gradients
    assert torch.isclose(dp_net.linear.weight.grad_sample, expect).all()
    dp_opt.step()

    # clip and sum
    expect = F.normalize(expect, dim=2).sum(axis=0, keepdim=False)
    # add noise + divide by expected batch size (3)
    expect = (
        expect +
        torch.normal(mean=0, std=1.0, size=(1,2), device=device, generator=torch.Generator(device).manual_seed(0))
    )/3
    # subtract from weights
    expect = torch.ones((1,2)).to(device) - expect
    assert torch.isclose(dp_net.linear.weight, expect).all()


def test_memory_manager_backward_below_threshold():
    ''' Test batch memory manager when the batch size is below threshold.
        Conclusion:
            - Taking backward (reduction=mean) on the mean loss results in per sample gradients
            - opt.step() will clip and sum, and divide by expected batch size
    '''

    device, _, _, _, _, _, dp_net, dp_opt, poisson_real_loader = opacus_setup(0.0, 10000.0)

    # fake data
    fake_x = torch.Tensor(
        [[10, 10],
         [9, 7],
         [11, 7]]
    ).float().to(device)

    max_physical_batch_size = 5 ## choose the max physical bsz

    fake_x_batches = []
    if max_physical_batch_size == 3:
        fake_x_batches = [fake_x[:2], fake_x[2:3]]
    elif max_physical_batch_size == 5:
        fake_x_batches = [fake_x]

    with dpgan.data.batch_memory_manager.BatchMemoryManager(
        data_loader=poisson_real_loader,
        max_physical_batch_size=max_physical_batch_size,
        optimizer=dp_opt
    ) as mem_poisson_real_loader:
        last_physical_batch = False
        for i, (x,) in enumerate(mem_poisson_real_loader):
            last_physical_batch = not dp_opt._step_skip_queue[-1] # pylint: disable=protected-access

            dp_opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                dp_net(torch.cat([x, fake_x_batches[i]])).squeeze(),
                torch.cat([torch.ones(len(x)), torch.zeros(len(fake_x_batches[i]))]).to(device),
                reduction = 'none'
            )

            # selected reduction = mean, take backward of mean loss to get correct per-example-grads
            loss.mean().backward()
            dp_opt.step()
            if last_physical_batch:
                break

    dp_opt.zero_grad()
    avg_grad = torch.Tensor([[-2, 21.5]]) # sum(per-example-grad)/8
    assert dp_net.linear.weight.detach().cpu().isclose(torch.Tensor([[1,1]]) - (avg_grad*8)/3).all()


def test_memory_manager_backward_above_threshold():
    ''' Test batch memory manager when the batch size is ABOVE threshold.
        Conclusion: result is the same as below threshold
                    - we should compute mean loss to backprop, compute a sum for logging purposes
    '''

    device, _, _, _, _, _, dp_net, dp_opt, poisson_real_loader = opacus_setup(0.0, 10000.0)

    # fake data
    fake_x = torch.Tensor(
        [[10, 10],
         [9, 7],
         [11, 7]]
    ).float().to(device)

    max_physical_batch_size = 3 ## choose the max physical bsz

    if max_physical_batch_size == 3:
        fake_x_batches = [fake_x[:2], fake_x[2:3]]
    elif max_physical_batch_size == 5:
        fake_x_batches = [fake_x]

    with dpgan.data.batch_memory_manager.BatchMemoryManager(
        data_loader=poisson_real_loader,
        max_physical_batch_size=max_physical_batch_size,
        optimizer=dp_opt
    ) as mem_poisson_real_loader:
        last_physical_batch = False
        for i, (x,) in enumerate(mem_poisson_real_loader):
            last_physical_batch = not dp_opt._step_skip_queue[-1] # pylint: disable=protected-access

            dp_opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                dp_net(torch.cat([x, fake_x_batches[i]])).squeeze(),
                torch.cat([torch.ones(len(x)), torch.zeros(len(fake_x_batches[i]))]).to(device),
                reduction = 'none'
            )

            # selected reduction = mean, take backward of mean loss to get correct per-example-grads
            loss.mean().backward()
            dp_opt.step()
            if last_physical_batch:
                break

    dp_opt.zero_grad()
    avg_grad = torch.Tensor([[-2, 21.5]]) # sum(per-example-grad)/8
    assert dp_net.linear.weight.detach().cpu().isclose(torch.Tensor([[1,1]]) - (avg_grad*8)/3).all()


def test_opacus_gnorm():
    ''' Test gnorm calculation when using a DP net
    '''

    device, _, _, _, _, _, dp_net, dp_opt, poisson_real_loader = opacus_setup(1.0, 1.0)

    # fake data
    fake_x = torch.Tensor(
        [[10, 10],
        [9, 7],
        [11, 7]]
    ).float().to(device)
    dp_opt.expected_batch_size =5

    x = None
    for i, (x,) in enumerate(poisson_real_loader):
        if i==0:
            continue
        dp_opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            dp_net(torch.cat([x, fake_x])).squeeze(),
            torch.cat([torch.ones(len(x)), torch.zeros(len(fake_x))]).to(device),
            reduction = 'none'
        )

        # selected reduction = mean, take backward of mean loss to get correct per-example-grads
        loss.mean().backward()
        break

    expect = torch.Tensor(
        [[9, -30],
         [9, -32]]
    ).float().to(device)
    assert (x == expect).all()

    # calc avg_grad, make sure its same for dp_net
    avg_grad = torch.cat([fake_x, -expect]).mean(dim=0)[None]
    assert torch.isclose(avg_grad,dp_net.linear.weight.grad).all()

    # make sure norm calc is correct
    expected_norm = torch.linalg.norm(avg_grad).item()
    assert abs(dpgan.utils.calc_gnorm(dp_net) - expected_norm) < 1e-5

    # make sure norm calc is different after step replaces .grad
    dp_opt.step()
    assert abs(dpgan.utils.calc_gnorm(dp_net) - expected_norm) > 1

if __name__ == '__main__':
    test_opacus_enable_hooks()
