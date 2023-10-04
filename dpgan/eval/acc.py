import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data

import dpgan.utils


class Net(torch.nn.Module):
    def __init__(self, img_dim = (1,28,28), num_labels=10):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(img_dim[0], 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)

        self.dropout1 = torch.nn.Dropout(.25)
        self.dropout2 = torch.nn.Dropout(0.5)

        if img_dim[1:] == (28,28):
            self.fc1 = torch.nn.Linear(9216, 128)
        elif img_dim[1:] == (32,32):
            self.fc1 = torch.nn.Linear(12544, 128)

        self.fc2 = torch.nn.Linear(128, num_labels)

    # forward method
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def calculate_acc(train_img, train_label, val_img, val_label, num_labels, device, seed,
                  bsz=64, num_epochs=10, lr=1e-3):

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_img[:-5000], train_label[:-5000]),
        batch_size=bsz,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    model_select_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_img[-5000:], train_label[-5000:]),
        batch_size=bsz,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed)
    )

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_img, val_label),
        batch_size=bsz,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed)
    )
    model = Net(img_dim=train_img.shape[1:], num_labels=num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f'| training CNN ({dpgan.utils.num_params(model)} params):')
    print(f'| n = {len(train_img)}, n_val = {len(val_img)}')
    print(f'| {num_epochs} epochs @ bsz = {bsz}, optim.Adam(lr={lr})')

    acc = train(model, train_loader, model_select_loader, val_loader, optimizer, num_epochs, device)
    return acc


def val(model, val_loader, device):

    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for _, (img, label) in enumerate(val_loader):
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            correct+= (torch.argmax(out, dim=1) == label).detach().sum().cpu().item()
            total += len(img)

    model.train()

    return correct/total


def train(model, train_loader, model_select_loader, val_loader, optimizer, num_epochs, device):

    model_select_accs = []
    val_accs = []

    for epoch in range (1, num_epochs+1):
        for i, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            out = model(img)
            loss = F.nll_loss(out, label)
            loss.backward()
            optimizer.step()

            if i%100 == 0:
                print(f'| epoch {epoch}, step {i}, loss: {loss.item()}')

        model_select_acc = val(model, model_select_loader, device)
        model_select_accs.append(model_select_acc)
        val_acc = val(model, val_loader, device)
        val_accs.append(val_acc)
        print(f'| model selection acc: {model_select_acc}')
        print(f'| val acc: {val_acc}')


    print(f'| model select accs: {model_select_accs}')
    print(f'| val accs:          {val_accs}')
    index = np.argmax(model_select_accs)

    print(f'| best model selection acc: {model_select_accs[index]} @ epoch {index+1}')
    val_acc_selected = val_accs[index]
    print(f'| val acc @ epoch {index+1}: {val_acc_selected}')

    return val_acc_selected
