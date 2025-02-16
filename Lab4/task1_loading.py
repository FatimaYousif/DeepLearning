from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision
import matplotlib.pyplot as plt


class MNISTMetricDataset(Dataset):
    def __init__(self, root="Lab1/FCNNs/mnist/", split='train', remove_class=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        # Task3e - classification of new classes
        if remove_class is not None:
            # -----filter out images with target class equal to remove_class
            mask = self.targets != remove_class
            self.images = self.images[mask]
            self.targets = self.targets[mask]

        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]
    
    # TODO
    # negative from the subset of images that do not share the anchor’s class.
    def _sample_negative(self, index):

        # target_id (8) = class label of anchor (8)
        target_id = self.targets[index].item()
        negative_indices = []
        for class_id in self.classes:
            if class_id != target_id:
                negative_indices.extend(self.target2indices[class_id])
        return choice(negative_indices)
    
    # TODO
    # positive from a subset of images that belong to the same class as the anchor
    def _sample_positive(self, index):
        target_id = self.targets[index].item()
        positive_indicies = self.target2indices[target_id]
        positive_indicies = [i for i in positive_indicies if i != index]
        return choice(positive_indicies)

    # anchor + positive + negative
    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    data = MNISTMetricDataset(remove_class=0)
    # print(data.target2indices[0])

    while True:
        random_idx = choice(range(len(data)))
        anchor, positive, negative, target_id = data[random_idx]
        # use matplotlib to show the images
        fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1)
        ax1.imshow(anchor.squeeze(0), cmap='gray')
        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow(positive.squeeze(0), cmap='gray')
        ax3 = fig.add_subplot(1,3,3)
        ax3.imshow(negative.squeeze(0), cmap='gray')
        plt.show()