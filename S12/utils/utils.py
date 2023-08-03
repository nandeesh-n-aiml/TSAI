import torch
import torch.optim as optim
from torch_lr_finder import LRFinder
import numpy as np
import math
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from lightning.fabric import Fabric
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt

cuda = torch.cuda.is_available()
fabric = Fabric(accelerator='cuda', precision="bf16-mixed")
fabric.launch()

def get_device() -> str:
    """ Get the device type. Returns CUDA/ CPU
    """
    return 'cuda' if cuda else 'cpu'

def set_seed(seed) -> None:
    """ Set seed for CPU and GPU
    
    Args:
        seed: a numeric seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def denormalise(tensor: torch.Tensor, mean: list, std: list):
    """ Denormalise an image tensor
    
    Args:
        tensor: An image tensor
        mean: Mean of the entire dataset
        std: Sandard deviation of the entire dataset
    """
    result = torch.tensor(tensor, requires_grad=False)
    for t, m, s in zip(result, mean, std):
        t.mul_(s).add_(m)
    return result

def get_optimizer(type, model, **kwargs):
    """ Get optimizer

    Args:
        type: Type of optimizer
        model: CNN model
        kwargs: key word args for optimizer
    """
    if type == 'SGD':
        return optim.SGD(model.parameters(), **kwargs)
    elif type == 'ADAM':
        return optim.Adam(model.parameters(), **kwargs)
    
def get_best_lr(model, train_loader, criterion, optimizer) -> float:
    """ Get max LR using `LRFinder` class.

    Args:
        model: CNN model
        train_loader: train data loader instance of `torch.utils.data.DataLoader`
        criterion: loss function
        optim_type: Type of optimizer to use
        kwargs: key word args for optimizer
    """
    lr_finder = LRFinder(model, optimizer, criterion, device=get_device())
    lr_finder.range_test(train_loader, end_lr=10, num_iter=100, step_mode="exp")
    _, max_lr = lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state
    return max_lr

def visualize_images(images: torch.Tensor,
                     labels: list,
                     mean: list,
                     std: list,
                     label_mapper: list=[],
                     n_cols: int=3,
                     figsize: tuple=(10,10),
                     img_title :str=None) -> None:
    """ Visualize images along with it's corresponding labels

    Args:
        images: A tensor of images
        labels: A tensor/list of labels corresponding to the images
        mean: Mean of the entire dataset
        std: Sandard deviation of the entire dataset
        label_mapper: list providing more intuitive information about labels. Ex: If labels are 1 & 0, label_mapper can be ['yes', 'no']
        n_cols: number of columns created for visualization in subplots
        figsize: figure size used for visualization
        img_title: image title
    """
    if type(labels) == torch.Tensor:
        labels = list(labels.numpy())
    mapper = {labels[i]: denormalise(images[i], mean, std) for i in range(images.shape[0])}
    mapper = dict(sorted(mapper.items(), key=lambda item: item[0]))
    plot_images(mapper, label_mapper, n_cols, figsize, img_title)

def visualize_imgs_with_gradcam(model,
                                images: torch.Tensor,
                                labels: list,
                                pred: list,
                                mean: list,
                                std: list,
                                n_cols: int=4,
                                figsize: tuple=(12, 9),
                                img_title :str=None) -> None:
    """ Visualize images using GradCAM library

    Args:
        model: A CNN based model. Ex: ResNet
        images: A tensor of images
        labels: A tensor/list of labels corresponding to the images
        pred: Model's prediction based on the images
        mean: Mean of the entire dataset
        std: Sandard deviation of the entire dataset
        n_cols: number of columns created for visualization in subplots
        figsize: figure size used for visualization
        img_title: image title
    """

    # Define the target layer for GradCAM
    target_layers = [model.layer3[-1]]
    # Create a GradCAM object
    gradcam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    mapper = {}
    for indx, img in enumerate(images):
        input_tensor = img.unsqueeze(0)
        targets = [ClassifierOutputTarget(pred[indx])]
        grayscale_cam = gradcam(input_tensor=input_tensor, targets=targets)
        dnorm_image = denormalise(img, mean, std)
        # Overlay the CAM on the original image
        cam_image = show_cam_on_image(dnorm_image.permute(1, 2, 0).numpy(), np.moveaxis(grayscale_cam, [0], [2]), use_rgb=True)
        mapper[labels[indx]] = dnorm_image
        mapper[str(indx + 1) + ' GradCAM'] = cam_image
    plot_images(mapper, n_cols=n_cols, figsize=figsize, img_title=img_title)
    plt.tight_layout()

def plot_images(mapper: dict, label_mapper: list=[], n_cols=3, figsize=(10,10), save_img=False, img_title=None) -> None:
    """ Plot images using matplotlib library

    Args:
        mapper: A mapper dictionary having label as a key and a corresponding image as the value
        label_mapper: list providing more intuitive information about labels. Ex: If labels are 1 & 0, label_mapper can be ['yes', 'no']
        n_cols: number of columns created for visualization in subplots
        figsize: figure size used for visualization
        img_title: image title
    """

    n_imgs = len(mapper.keys())
    row_indx = -1
    n_rows = math.ceil(n_imgs/n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    for indx, lbl in enumerate(mapper):
        if indx%n_cols == 0:
            row_indx += 1
        if type(mapper[lbl]) == torch.Tensor:
            img = mapper[lbl].permute(1,2,0)
            img = img.numpy()
        else:
            img = mapper[lbl]
        
        if save_img:
            min_, max_ = img.min(), img.max()
            img = (img - min_)/(max_ - min_)
            try:
                img_text = str(lbl)+'.png'
                img_text = img_text.replace(':', '=').replace('\n', '; ')
                plt.imsave(img_text, img)
            except Exception as e:
                print('Save error:', lbl, img.shape, img.min(), img.max(), e)
        
        axs[row_indx][indx%n_cols].imshow(img)
        if len(label_mapper) == 0:
            axs[row_indx][indx%n_cols].set_title(lbl)
        else:
            axs[row_indx][indx%n_cols].set_title(label_mapper[lbl] + f' ({lbl})')
    if type(img_title) is not None:
        fig.suptitle(img_title)
