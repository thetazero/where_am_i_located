from torchvision import transforms, utils, io
import os
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt


class CampusImagesDataSet(Dataset):

    def __init__(self, root_dir, transform=None):
        self.labels = json.load(open(f'{root_dir}/labels.json'))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels.keys())

    def __getitem__(self, idx):
        image_name = list(self.labels.keys())[idx]
        image = io.read_image(f'{self.root_dir}/{image_name}')

        label = self.labels[image_name]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == "__main__":
    dataset = CampusImagesDataSet('processed_data')
    for i in range(4):
        sample = dataset[i]
        print(i, sample['image'].shape, sample['label'])
        plt.imshow(sample['image'].permute(1, 2, 0))
        plt.show()
