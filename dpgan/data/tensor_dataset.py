import torch.utils.data
import torchvision.transforms

class TensorDataset(torch.utils.data.TensorDataset):
    def __init__(self, *tensors: torch.Tensor, transform = torchvision.transforms.Compose([])):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, index):
        return ((self.transform(self.tensors[0][index]),) +
                tuple(self.tensors[j][index] for j in range(1, len(self.tensors))))
