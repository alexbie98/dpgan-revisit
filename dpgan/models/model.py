import torch
from torch import nn
import torch.nn.functional as F

def get_label_to_z_label(num_labels):
    label_to_z_label = F.one_hot(
        torch.arange(num_labels),
        num_labels
    ).unsqueeze(-1).unsqueeze(-1).float()
    return label_to_z_label

def get_label_to_img_label(img_dim, num_labels):
    label_to_img_label = get_label_to_z_label(num_labels).repeat((1, 1) + img_dim[1:])
    return label_to_img_label

class Generator(nn.Module):

    def __init__(self, dim_latent):
        super().__init__()
        self.dim_latent = dim_latent

    def sample(self, bsz, device):
        z = torch.randn((bsz, self.dim_latent)).unsqueeze(-1).unsqueeze(-1).to(device)
        return self(z), torch.zeros((1,)).long().to(device)

    forward = nn.Module.forward

class LabelledGenerator(Generator):

    def __init__(self, dim_latent, num_labels, device):
        super().__init__(dim_latent)
        self.num_labels = num_labels
        self.label_to_z_label = get_label_to_z_label(num_labels).to(device)

    def set_device(self, device):
        self.label_to_z_label = self.label_to_z_label.to(device)

    def sample(self, bsz, device):
        z = torch.randn((bsz, self.dim_latent)).unsqueeze(-1).unsqueeze(-1).to(device)
        fake_label = torch.randint(0, self.num_labels, (bsz,)).to(device).detach()
        return self(z, self.label_to_z_label[fake_label]), fake_label

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.img_dim = img_dim

    forward = nn.Module.forward

class LabelledDiscriminator(Discriminator):
    def __init__(self, img_dim, num_labels, device):
        super().__init__(img_dim)
        self.label_to_img_label = get_label_to_img_label(img_dim, num_labels).to(device)

    def set_device(self, device):
        self.label_to_img_label = self.label_to_img_label.to(device)
