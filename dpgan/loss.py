import torch.nn.functional as F

def nonsaturating_logistic(fake_score):
    return -F.logsigmoid(fake_score)
