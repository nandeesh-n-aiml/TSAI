import torch
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

class Scheduler:
    """ LR scheduler class compatible for all pytorch schedulers from `torch.optim`.
    """

    def __init__(self, type: str, kwargs: dict):
        """ Constructor to initialize scheduler

        Args:
            type: scheduler type
            kwargs: keyword arguments required to initialize the scheduler
        """
        self.lr_change = []
        self.scheduler_type = type
        self.scheduler = getattr(lr_scheduler, type)(**kwargs)

    def step(self):
        """ Update LR of the optimizer and track it for visualization
        """
        self.scheduler.step()
        self.lr_change.append(self.scheduler.get_last_lr()[0])

    def plot_lr_change(self):
        """ Visualize change in LR over step/epoch
        """
        if self.scheduler_type == 'OneCycleLR':
            xlabel = 'Step(s)'
        else:
            xlabel = 'Epoch(s)'
        plt.plot(self.lr_change)
        plt.xlabel(xlabel)
        plt.ylabel('LR')
        plt.title('LR vs ' + xlabel)
