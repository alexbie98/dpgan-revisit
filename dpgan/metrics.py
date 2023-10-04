
import io
import matplotlib.pyplot as plt
import PIL
import numpy as np

import torchvision.transforms


def d_metrics_init():
    return {
        'd_loss':                   (0, 0), # mean vars (accum value, accum count)
        'd_loss/real':              (0, 0),
        'd_loss/fake':              (0, 0),
        'd_acc':                    (0, 0),
        'd_acc/real':               (0, 0),
        'd_acc/fake':               (0, 0),
        'bsz/real_mean_physical':   (0, 0),
        'bsz':                      (0, 1), # sum vars (accum value, 1),
        'bsz/fake':                 (0, 1),
        'bsz/real':                 (0, 1),
    }

def accumulate(old_m, update):
    assert old_m.keys() == update.keys()

    new_m = {}
    for k in old_m.keys():
        new_m[k] = (old_m[k][0] + update[k][0], old_m[k][1] + update[k][1])

    return new_m

def publish(m, writer, d_step):
    for k in m.keys():
        step = d_step-1 if k.startswith('d') else d_step
        amount = m[k][0]/m[k][1]
        if 'norm' in k:
            amount = amount.norm()
        writer.add_scalar(k, amount, step)


def create_example_img(img, label, display_labels):

    num_samples = len(img)
    assert num_samples % 5 == 0

    fig, axes = plt.subplots(num_samples//5, 5)
    fig.tight_layout(pad=0)
    fig.subplots_adjust(hspace=0.45, wspace=0.00)

    # plot
    for i in range(int(num_samples/5)):
        for j in range(5):
            img_ij = np.transpose(img[i*5+j], (1,2,0))
            axes[i, j].imshow(img_ij[:,:,0] if img_ij.shape[2] == 1 else img_ij, cmap='gray', interpolation='none')
            axes[i, j].set_xlabel(display_labels[label[i*5+j].item()])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    # to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    plot = PIL.Image.open(buf)

    # to tensor
    transform = torchvision.transforms.ToTensor()
    plot = transform(plot)
    buf.close()

    return plot
