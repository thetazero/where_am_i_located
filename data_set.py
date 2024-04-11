from torchvision import transforms, utils, io
import os
from torch.utils.data import Dataset, DataLoader
import torch
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def normalize_point(point, mean, std):
    return (point - mean) / std


def denormalize_point(point, mean, std):
    return point * std + mean


class CampusImagesDataSet(Dataset):

    def __init__(self, root_dir, transform=transforms.ToTensor()):
        self.labels = json.load(open(f'{root_dir}/labels.json'))
        self.root_dir = root_dir
        self.transform = transform

        self.latitudes = [p[0] for p in self.labels.values()]
        self.longitudes = [p[1] for p in self.labels.values()]
        self.mean_latitude = sum(self.latitudes) / len(self.latitudes)
        self.mean_longitude = sum(self.longitudes) / len(self.longitudes)
        self.std_latitude = np.std(self.latitudes)
        self.std_longitude = np.std(self.longitudes)
    
    def label_to_real_location(self, label):
        lat, lon = label
        return denormalize_point(lat, self.mean_latitude, self.std_latitude), denormalize_point(lon, self.mean_longitude, self.std_longitude)

    def __len__(self):
        return len(self.labels.keys())

    def __getitem__(self, idx):
        image_name = list(self.labels.keys())[idx]
        image = Image.open(f'{self.root_dir}/{image_name}')

        if self.transform:
            image = self.transform(image)

        label = self.labels[image_name]
        [lat, lon] = label
        lat = normalize_point(lat, self.mean_latitude, self.std_latitude)
        lon = normalize_point(lon, self.mean_longitude, self.std_longitude)
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
