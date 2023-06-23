import torch
import math
import matplotlib.pyplot as plt

cuda = torch.cuda.is_available()

def get_device():
    return 'cuda' if cuda else 'cpu'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def visualize_images(images, labels, label_mapper=[], n_cols=3, figsize=(10,10), img_title=None):
    if type(labels) == torch.Tensor:
        labels = list(labels.numpy())
    mapper = {labels[i]: images[i] for i in range(images.shape[0])}
    mapper = dict(sorted(mapper.items(), key=lambda item: item[0]))
    
    # Plot images
    n_imgs = len(mapper.keys())
    row_indx = -1
    n_rows = math.ceil(n_imgs/n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    for indx, lbl in enumerate(mapper):
        if indx%n_cols == 0:
            row_indx += 1
        axs[row_indx][indx%n_cols].imshow(mapper[lbl].permute(1,2,0))
        if len(label_mapper) == 0:
            axs[row_indx][indx%n_cols].set_title(lbl)
        else:
            axs[row_indx][indx%n_cols].set_title(label_mapper[lbl] + f' ({lbl})')
    if type(img_title) is not None:
        fig.suptitle(img_title)
