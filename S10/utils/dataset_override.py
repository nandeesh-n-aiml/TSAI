from torchvision.datasets import CIFAR10
from PIL import Image
from typing import Any, Tuple

class CIFAR10_Override(CIFAR10):
    """ Overriding class CIFAR10 to support `albumentations` library
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform = None,
        target_transform = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            if self.trans_lib == 'torchvision':
                # doing this so that it is consistent with all other datasets
                # to return a PIL Image
                img = Image.fromarray(img)
                img = self.transform(img)
            elif self.trans_lib == 'albumentations':
                img = self.transform(image=img)['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
