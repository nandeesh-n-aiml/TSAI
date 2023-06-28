from torchvision import transforms

class DataTransform:
    """ Implements data transformations required to be applied for a dataset
    """
    
    def __init__(self):
        """ Initialize the data transform without transformations and normalization as the mean and std values are unknown.
        """
        self.set_transforms('train', [], False)
        self.test_transforms = False

    def set_mean(self, mean: float):
        """ Set mean

        Args:
            mean: a floating point number
        """
        self.mean = mean
        return self

    def set_std(self, std: float):
        """ Set standard deviation

        Args:
            std: a floating point number
        """
        self.std = std
        return self

    def get_transforms(self) -> tuple:
        """ Get train and test transforms
        """
        return (self.train_transforms, self.test_transforms)

    def set_transforms(self, dataset_type: str, list_of_transforms: list, normalize: bool=True):
        """ Set train or test transform on a dataset

        Args:
            dataset_type: Accepts either "train" or "test"
            list_of_transforms: A list of transformation needs to be applied to an image
            normalize: Flag that indicates if the dataset needs to be normalized or not
        """
        list_of_transforms.append(transforms.ToTensor())
        if normalize:
            list_of_transforms.append(transforms.Normalize(self.mean, self.std))
        if dataset_type == 'train':
            self.train_transforms = transforms.Compose(list_of_transforms)
        elif dataset_type == 'test':
            self.test_transforms = transforms.Compose(list_of_transforms)
        return self
