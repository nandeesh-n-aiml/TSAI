import torch
from torchvision import datasets
from . import data_transforms
from . import utils

class Dataset:
    """ Implements pytorch compatible datasets
    """

    def __init__(self, dataset_name: str, normalize: bool=True):
        """ Constructor to initialize dataset

        Args:
            dataset_name: Dataset name. Ex: MNIST, CIFAR10 etc.
            normalize: Flag that indicates if the dataset needs to be normalized or not
        """
        self.dataset_name = dataset_name
        self.dt = data_transforms.DataTransform()
        self.set_dataloader_cfgs()
        self.set_train_data()
        self.set_test_data()
        if normalize:
            self.normalize_dataset()

    def set_dataloader_cfgs(self):
        """ Set configs for `torch.utils.data.DataLoader` instance based on the device (CPU/CUDA).
        """
        dataloader_cfgs = {
            'batch_size': 128,
            'shuffle': True
        }
        if utils.cuda:
            dataloader_cfgs['num_workers'] = 4
            dataloader_cfgs['pin_memory'] = True
        self.dataloader_cfgs = dataloader_cfgs
        return self
    
    def get_moments(self) -> tuple:
        """ Get mean and SD for a given dataset.
        """
        if self.dataset_name == 'MNIST':
            train_data = self.train.transform(self.train.data.numpy())
            mean, std = (torch.mean(train_data).tolist(), ), (torch.std(train_data).tolist(), )
        elif self.dataset_name == 'CIFAR10':
            train_data = self.train.data/255
            mean, std = train_data.mean(axis=(0,1,2)), train_data.std(axis=(0,1,2))
        print(f'The mean and SD for {self.dataset_name} dataset are {mean} and {std} respectively.')
        return (mean, std)

    def set_train_data(self):
        """ Initialize train dataset and sets the reference to `train` property.
        Downloads the dataset if its unavailable in the local directory.
        """
        self.train = getattr(datasets, self.dataset_name)('../data', train=True, download=True, transform=self.dt.train_transforms)
        return self

    def set_test_data(self):
        """ Initialize test dataset and sets the reference to `test` property.
        Downloads the dataset if its unavailable in the local directory.
        """
        self.test = getattr(datasets, self.dataset_name)('../data', train=False, download=True, transform=self.dt.test_transforms)
        return self

    def normalize_dataset(self) -> None:
        """ Normalize dataset based on the mean and SD computed for a dataset.
        """
        mean, std = self.get_moments()
        mean, std = tuple(mean), tuple(std)
        self.dt \
            .set_mean(mean) \
            .set_std(std) \
            .set_transforms('train', [], normalize=True) \
            .set_transforms('test', [], normalize=True)
        self.set_train_data() \
            .set_test_data()

    def get_data_loaders(self, reset=True) -> tuple:
        """ Get train & test dataloaders

        Args:
            reset: Initialize train and test datasets
        """
        if reset:
            self.set_train_data() \
                .set_test_data()
        train_loader = torch.utils.data.DataLoader(self.train, **self.dataloader_cfgs)
        test_loader = torch.utils.data.DataLoader(self.test, **self.dataloader_cfgs)
        return (train_loader, test_loader)
