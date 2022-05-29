from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class DataSet:
    def __init__(self,root_dir,img_size):
        self.root_dir=root_dir
        self.img_size=img_size
        self.data_transforms = transforms.Compose([
            transforms.Resize(size=(self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std = [0.5, 0.5, 0.5])
        ])

    def dataset(self):
        return ImageFolder(root=self.root_dir, transform=self.data_transforms)


