# Library imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from torchinfo import summary
import matplotlib.pyplot as plt
from utils.utils import fabric

class ModelComposite(LightningModule):
    """ This class is created based on the composite design principle. It contains all the common methods related to the model.
    """

    def __init__(self):
        """ Constructor to initialize train and test related properties
        """
        super(ModelComposite, self).__init__()
        self.train_accuracy = []
        self.val_accuracy = []
        self.train_losses = []
        self.val_losses = []
        self.lr_change = []
        self.norm_type = 'bn'
        self.n_groups = 2
        self.max_lr = 0.01

    def print_summary(self, input_size: tuple):
        """ Print model summary

        Args:
            input_size: Input image size in (N, C, H, W)
        """
        summary(self, input_size, verbose=1)

    def get_norm(self, n_channels: int):
        """ Get a normalization layer. It could be one of the following:
            - Batch normalization
            - Layer normalization
            - Group normalization

        Args:
            n_channels: number of channels to create a normalization layer
        """
        if self.norm_type == 'bn':
            return nn.BatchNorm2d(n_channels)
        elif self.norm_type == 'ln':
            return nn.GroupNorm(1, n_channels)
        elif self.norm_type == 'gn':
            return nn.GroupNorm(self.n_groups, n_channels)

    def plot_accuracy(self) -> None:
        """
        Visualize model train and test accuracy over n_epochs
        """
        epochs = list(range(1, len(self.train_accuracy) + 1))
        plt.plot(epochs, self.train_accuracy, label='Train accuracy')
        plt.plot(epochs, self.val_accuracy, label='Test accuracy')
        plt.legend()

    def plot_loss(self) -> None:
        """
        Visualize model train and test losses over n_epochs
        """
        epochs = list(range(1, len(self.train_accuracy) + 1))
        plt.plot(epochs, self.train_losses, label='Train loss')
        plt.plot(epochs, self.val_losses, label='Test loss')
        plt.legend()

    def get_incorrect_pred(self, device: str, test_loader, top_n: int=10) -> tuple:
        """ Get top_n incorrect model predictions from test set.

        Args:
            device: CUDA/CPU
            test_loader: test data loader instance of `torch.utils.data.DataLoader`
            top_n: number of incorrect predictions to return
        """
        with torch.no_grad():
            data, target = next(iter(test_loader))
            data, target = data.to(device), target.to(device)
            y_pred = self(data)
            pred = y_pred.argmax(dim=1)
            compare = pred.eq(target)
            incorrect_indx = torch.where((compare == False), 1, 0).nonzero()
            top_n_pred = incorrect_indx[:top_n].squeeze()
            return (data[top_n_pred], target[top_n_pred], pred[top_n_pred])

    # Lightning module methods
    def configure_optimizers(self):
        """ Return optimizer, scheduler
        """
        return [self.optimizer], [{
            'scheduler': self.scheduler,
            'interval': 'step',
            'frequency': 1
        }]

    def training_step(self, batch, batch_idx):
        """ Model training per batch

        Args:
            batch: Batch of images and labels
            batch_idx: Batch index
        """
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        accuracy = correct/len(x)
        self.lr_change.append(self.scheduler.get_last_lr()[0])
        self.log_dict({"train_acc": accuracy, "train_loss": loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """ Model validation per batch

        Args:
            batch: Batch of images and labels
            batch_idx: Batch index
        """
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        accuracy = correct/len(x)
        self.log_dict({"val_acc": accuracy, "val_loss": loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True)

class MetricTracker(Callback):
    """ This class tracks the metrics related to the model. It is a part of callbacks for pytorch lightning.
    """

    def __init__(self):
        """ Constructor to initialize tracker class
        """
        self.val_loss = []
        self.val_acc = []
        self.lr = []
        
    def on_train_epoch_end(self, trainer, model):
        """ A callback hook to perform certain action at the end of training epoch.
        """
        train_loss = trainer.logged_metrics['train_loss'].item()
        train_acc = trainer.logged_metrics['train_acc'].item()
        model.train_losses.append(train_loss)
        model.train_accuracy.append(train_acc)
        lr = model.optimizers().param_groups[0]['lr']
        self.lr.append(lr)

    def on_validation_epoch_end(self, trainer, model):
        """ A callback hook to perform certain action at the end of validation epoch.
        """
        val_loss = trainer.logged_metrics['val_loss'].item()
        val_acc = trainer.logged_metrics['val_acc'].item()
        model.val_losses.append(val_loss)
        model.val_accuracy.append(val_acc)
        torch.save(model.state_dict(), f'./saved_models/model_{trainer.current_epoch}_val_acc_{round(val_acc, 3)}.pth')
