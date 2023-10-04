import torch
from torch import nn
import torch.nn.functional as F

from .model import LabelledGenerator, LabelledDiscriminator


# MNIST DCGAN -----------------------------------------------------------------

class DCLabelledGenerator(LabelledGenerator):
    def __init__(self, dim, img_dim, dim_latent, num_labels, device):
        assert img_dim == (1,28,28)
        super().__init__(dim_latent, num_labels, device)

        self.deconv1_1 = nn.ConvTranspose2d(self.dim_latent, dim*2, 4, 1, 0)  # z -> [2d x H x W]
        self.deconv1_2 = nn.ConvTranspose2d(self.num_labels, dim*2, 4, 1, 0)  # y -> [2d x H x W]

        self.deconv2 = nn.ConvTranspose2d(dim*4, dim*2, 3, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(dim*2, dim, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(dim, 1, 4, 2, 1)

    def forward(self, z, y):
        x = F.leaky_relu(self.deconv1_1(z), 0.2)
        y = F.leaky_relu(self.deconv1_2(y), 0.2)
        x = torch.cat([x, y], 1)

        x = F.leaky_relu(self.deconv2(x), 0.2)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        x = (torch.tanh(self.deconv4(x)) + 1) * 0.5  # [-1,1] -> [0,1]
        return x


class DCLabelledDiscriminator(LabelledDiscriminator):
    def __init__(self, dim, img_dim, num_labels, device):
        assert img_dim == (1,28,28)
        super().__init__(img_dim, num_labels, device)

        self.conv1_1 = nn.Conv2d(1, dim//2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(num_labels, dim//2, 4, 2, 1)

        self.conv2 = nn.Conv2d(dim, dim*2, 4, 2, 1)
        self.conv3 = nn.Conv2d(dim*2, dim*4, 3, 2, 1)
        self.conv4 = nn.Conv2d(dim*4, 1, 4, 1, 0)

    def forward(self, x, label):
        x = F.leaky_relu(self.conv1_1(x), 0.2)
        y = F.leaky_relu(
            self.conv1_2(self.label_to_img_label[label]), 0.2
        )
        x = torch.cat([x, y], 1)

        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.conv4(x)
        return x.squeeze()


# CELEBA DCGAN ----------------------------------------------------------------

class DC32LabelledGenerator(LabelledGenerator):
    def __init__(self, dim, img_dim, dim_latent, num_labels, device):
        assert img_dim == (3,32,32)
        super().__init__(dim_latent, num_labels, device)

        self.deconv1_1 = nn.ConvTranspose2d(self.dim_latent, dim*4, 4, 2, 1)    # z -> [4d x 2 x 2]
        self.deconv1_2 = nn.ConvTranspose2d(self.num_labels, dim*4, 4, 2, 1)    # y -> [4d x 2 x 2]
        self.deconv2 = nn.ConvTranspose2d(dim*8, dim*4, 4, 2, 1)    # [8d x 2 x 2] -> [4d x 4 x 4]
        self.deconv3 = nn.ConvTranspose2d(dim*4, dim*2, 4, 2, 1)    # [4d x 4 x 4] -> [2d x 8 x 8]
        self.deconv4 = nn.ConvTranspose2d(dim*2, dim, 4, 2, 1)      # [2d x 8 x 8] -> [d x 16 x 16]
        self.deconv5 = nn.ConvTranspose2d(dim, img_dim[0], 4, 2, 1) # [d x 16 x 16] -> [3 x 32 x 32]

    def forward(self, z, y):
        x = F.leaky_relu(self.deconv1_1(z), 0.2)
        y = F.leaky_relu(self.deconv1_2(y), 0.2)
        x = torch.cat([x, y], 1)

        x = F.leaky_relu(self.deconv2(x), 0.2)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        x = F.leaky_relu(self.deconv4(x), 0.2)
        x = (torch.tanh(self.deconv5(x)) + 1) * 0.5  # [-1,1] -> [0,1]
        return x


class DC32LabelledDiscriminator(LabelledDiscriminator):
    def __init__(self, dim, img_dim, num_labels, device):
        assert img_dim == (3,32,32)
        super().__init__(img_dim, num_labels, device)

        self.conv1_1 = nn.Conv2d(img_dim[0], dim//2, 4, 2, 1)   # x -> [d/2 x 16 x 16]
        self.conv1_2 = nn.Conv2d(num_labels, dim//2, 4, 2, 1)   # y -> [d/2 x 16 x 16]

        self.conv2 = nn.Conv2d(dim, dim*2, 4, 2, 1)     # [d x 16 x 16] -> [2d x 8 x 8]
        self.conv3 = nn.Conv2d(dim*2, dim*4, 4, 2, 1)   # [2d x 8 x 8] -> [4d x 4 x 4]
        self.conv4 = nn.Conv2d(dim*4, dim*8, 4, 2, 1)   # [4d x 4 x 4] -> [8d x 2 x 2]
        self.conv5 = nn.Conv2d(dim*8, 1, 4, 2, 1)       # [8d x 2 x 2] -> [1 x 1 x 1]

    def forward(self, x, label):
        x = F.leaky_relu(self.conv1_1(x), 0.2)
        y = F.leaky_relu(
            self.conv1_2(self.label_to_img_label[label]), 0.2
        )
        x = torch.cat([x, y], 1)

        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.conv5(x)
        return x.squeeze()


# CELEBA SMALL DCGAN ----------------------------------------------------------

class DC32SmallLabelledGenerator(LabelledGenerator):
    def __init__(self, dim, img_dim, dim_latent, num_labels, device):
        assert img_dim == (3,32,32)
        super().__init__(dim_latent, num_labels, device)

        self.deconv1_1 = nn.ConvTranspose2d(self.dim_latent, dim*2, 4, 1, 0)    # z -> [2d x 4 x 4]
        self.deconv1_2 = nn.ConvTranspose2d(self.num_labels, dim*2, 4, 1, 0)    # y -> [2d x 4 x 4]
        self.deconv2 = nn.ConvTranspose2d(dim*4, dim*2, 4, 2, 1)    # [4d x 4 x 4] -> [2d x 8 x 8]
        self.deconv3 = nn.ConvTranspose2d(dim*2, dim, 4, 2, 1)      # [2d x 8 x 8] -> [d x 16 x 16]
        self.deconv4 = nn.ConvTranspose2d(dim, img_dim[0], 4, 2, 1) # [d x 16 x 16] -> [3 x 32 x 32]

    def forward(self, z, y):
        x = F.leaky_relu(self.deconv1_1(z), 0.2)
        y = F.leaky_relu(self.deconv1_2(y), 0.2)
        x = torch.cat([x, y], 1)

        x = F.leaky_relu(self.deconv2(x), 0.2)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        x = (torch.tanh(self.deconv4(x)) + 1) * 0.5  # [-1,1] -> [0,1]
        return x


class DC32SmallLabelledDiscriminator(LabelledDiscriminator):
    def __init__(self, dim, img_dim, num_labels, device):
        assert img_dim == (3,32,32)
        super().__init__(img_dim, num_labels, device)

        self.conv1_1 = nn.Conv2d(img_dim[0], dim//2, 4, 2, 1)   # x -> [d/2 x 16 x 16]
        self.conv1_2 = nn.Conv2d(num_labels, dim//2, 4, 2, 1)   # y -> [d/2 x 16 x 16]

        self.conv2 = nn.Conv2d(dim, dim*2, 4, 2, 1)     # [d x 16 x 16] -> [2d x 8 x 8]
        self.conv3 = nn.Conv2d(dim*2, dim*4, 4, 2, 1)   # [2d x 8 x 8] -> [4d x 4 x 4]
        self.conv4 = nn.Conv2d(dim*4, 1, 4, 1, 0)       # [4d x 4 x 4] -> [1 x 1 x 1]

    def forward(self, x, label):
        x = F.leaky_relu(self.conv1_1(x), 0.2)
        y = F.leaky_relu(
            self.conv1_2(self.label_to_img_label[label]), 0.2
        )
        x = torch.cat([x, y], 1)

        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.conv4(x)
        return x.squeeze()
