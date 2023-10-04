import math
import torch

def num_params(model):
    return sum(math.prod(x.size()) for x in model.parameters() if x.requires_grad)

def count_split(count, sections):
    '''
    Given `count` to be split among `sections` partitions,
    split them close to equally, with the ones with extra entries at the start.
    Should match behaviour of np.array_split
    '''
    base_amount = count // sections
    plus_one_sections = [base_amount+1]*(count%sections)
    base_amount_sections = [base_amount] * (sections - (count%sections))
    return plus_one_sections + base_amount_sections

def calc_gnorm(model):
    '''
    Given a model, calculate the gradient norm
    '''
    with torch.no_grad():
        gnorm_sq = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)
    return torch.sqrt(gnorm_sq)

def get_gvec(model):
    '''
    Given a model, calculate the flattened vector of all its gradients
    '''
    with torch.no_grad():
        gvec = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return gvec

def weights_zeros_like(model):
    '''
    Given a model, get its weights, but 0'd out
    '''
    with torch.no_grad():
        weights = [torch.zeros(p.data.shape, requires_grad=False).to(p.device)
                for p in model.parameters()]
    return weights

def weights_copy(model):
    '''
    Given a model, get its weights.
    '''
    weights = weights_zeros_like(model)

    with torch.no_grad():
        for (weight, model_p) in zip(weights, model.parameters()):
            weight.copy_(model_p.data)
    return weights

def weights_add_scaled(weights, model, scale):
    '''
    set weights = weights + scale * model
    '''
    with torch.no_grad():
        for (weight, model_p) in zip(weights, model.parameters()):
            weight.add_(scale * model_p.data)

def weights_scale(weights, scale):
    '''
    set weights = scale * weights
    '''
    with torch.no_grad():
        for weight in weights:
            weight.mul_(scale)

def weights_assign_model(weights, model):
    '''
    assigns to model weights. Note that model will now point to weights, so we should stop using that reference
    '''
    with torch.no_grad():
        for (weight, model_p) in zip(weights, model.parameters()):
            model_p.data = weight


def round_normalized_img(img, buckets=255):
    return (img * buckets).round()/buckets
