from torchvision import transforms

class DataTransform:
    def __init__(self):
        # Initially the mean and std values are unknown.
        self.set_transforms('train', [], False)
        self.test_transforms = False

    def set_mean(self, mean):
        self.mean = mean
        return self

    def set_std(self, std):
        self.std = std
        return self

    def get_transforms(self):
        return (self.train_transforms, self.test_transforms)

    def set_transforms(self, dataset_type, list_of_transforms:list, normalize:bool=True):
        list_of_transforms.append(transforms.ToTensor())
        if normalize:
            list_of_transforms.append(transforms.Normalize(self.mean, self.std))
        if dataset_type == 'train':
            self.train_transforms = transforms.Compose(list_of_transforms)
        elif dataset_type == 'test':
            self.test_transforms = transforms.Compose(list_of_transforms)
        return self
