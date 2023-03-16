import torch
from torch.utils.data import Dataset
class datasetCFAR10(Dataset):

    def __init__(self, data, transform=None):
        self.train = torch.Tensor(data['images'])
        self.labels = torch.Tensor(data['labels'])
        self.transform = transform

    def __len__(self):
        return len(self.train)

    def __getitem__(self, index):
        image = self.train[index].reshape(3, 32, 32)
        # image = image.transpose(1, 2, 0)
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label