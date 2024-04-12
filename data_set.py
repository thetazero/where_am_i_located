from torchvision import transforms, utils, io
import os
from torch.utils.data import Dataset, DataLoader
import torch
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def normalize_point(point, minval, maxval):
    return (point - minval) / (maxval - minval)


def denormalize_point(point, minval, maxval):
    return point * (maxval - minval) + minval


class CampusImagesDataSet(Dataset):

    def __init__(self, root_dir, transform=transforms.ToTensor()):
        self.labels = json.load(open(f'{root_dir}/labels.json'))
        self.root_dir = root_dir
        self.transform = transform

        self.latitudes = [p[0] for p in self.labels.values()]
        self.longitudes = [p[1] for p in self.labels.values()]

        self.min_lat = min(self.latitudes)
        self.max_lat = max(self.latitudes)
        self.min_lon = min(self.longitudes)
        self.max_lon = max(self.longitudes)
    
    def label_to_real_location(self, label):
        lat, lon = label
        return denormalize_point(lat, self.min_lat, self.max_lat), denormalize_point(lon, self.min_lon, self.max_lon)

    def __len__(self):
        return len(self.labels.keys())

    def __getitem__(self, idx):
        image_name = list(self.labels.keys())[idx]
        image = Image.open(f'{self.root_dir}/{image_name}')

        if self.transform:
            image = self.transform(image)

        label = self.labels[image_name]
        [lat, lon] = label
        lat = normalize_point(lat, self.min_lat, self.max_lat)
        lon = normalize_point(lon, self.min_lon, self.max_lon)
        label = torch.from_numpy(
            np.array([lat, lon], dtype=np.float32)
        )
        sample = [image, label]

        return sample


if __name__ == "__main__":
    dataset = CampusImagesDataSet('processed_data')
    for i in range(4):
        sample = dataset[i]
        print(i, sample['image'].shape, sample['label'])
        plt.imshow(sample['image'].permute(1, 2, 0))
        plt.show()
