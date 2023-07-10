import torch
from torchvision import datasets
from .data_transforms import DataTransform
from . import utils
from .dataset_override import CIFAR10_Override

class Dataset:
    """ Implements pytorch compatible datasets
    """
    dataset_fn_mapper = {
        'MNIST': datasets.MNIST,
        'CIFAR10': CIFAR10_Override
    }

    def __init__(self,
                 dataset_name: str,
                 normalize: bool=True,
                 batch_size: int=128,
                 trans_lib: str='torchvision'):
        """ Constructor to initialize dataset

        Args:
            dataset_name: Dataset name. Ex: MNIST, CIFAR10 etc.
            normalize: Flag that indicates if the dataset needs to be normalized or not
            batch_size: Batch size. Defaults to 128
        """
        self.dataset_name = dataset_name
        self.trans_lib = trans_lib
        self.batch_size = batch_size
        self.dt = DataTransform(trans_lib)
        self.set_dataset_fn()
        self.set_train_data()
        self.set_test_data()
        if normalize:
            self.normalize_dataset()

    def set_dataset_fn(self):
        """ Setter method to set the dataset function to fetch data
        """
        self.dataset_fn = self.dataset_fn_mapper.get(self.dataset_name)

    def set_batch_size(self, batch_size: int):
        """ Setter method to set `batch_size`

        Args:
            batch_size: Batch size
        """
        self.batch_size = batch_size
        return self

    def get_dataloader_cfgs(self):
        """ Get configs for `torch.utils.data.DataLoader` instance based on the device (CPU/CUDA).
        """
        dataloader_cfgs = {
            'batch_size': self.batch_size,
            'shuffle': True
        }
        if utils.cuda:
            dataloader_cfgs['num_workers'] = 4
            dataloader_cfgs['pin_memory'] = True
        return dataloader_cfgs
    
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
        self.train = self.dataset_fn('../data', train=True, download=True, transform=self.dt.train_transforms)
        self.train.trans_lib = self.trans_lib
        return self

    def set_test_data(self):
        """ Initialize test dataset and sets the reference to `test` property.
            Downloads the dataset if its unavailable in the local directory.
        """
        self.test = self.dataset_fn('../data', train=False, download=True, transform=self.dt.test_transforms)
        self.test.trans_lib = self.trans_lib
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
        dataloader_cfgs = self.get_dataloader_cfgs()
        train_loader = torch.utils.data.DataLoader(self.train, **dataloader_cfgs)
        test_loader = torch.utils.data.DataLoader(self.test, **dataloader_cfgs)
        return (train_loader, test_loader)
