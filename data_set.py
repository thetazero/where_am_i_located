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


def into_onehot(lat, lon, min_lat, max_lat, min_lon, max_lon, grid_size):
    lat = int(np.floor(normalize_point(lat, min_lat, max_lat) * grid_size))
    lon = int(np.floor(normalize_point(lon, min_lon, max_lon) * grid_size))
    lat = np.clip(lat, 0, grid_size - 1)
    lon = np.clip(lon, 0, grid_size - 1)

    assert 0 <= lat < grid_size
    assert 0 <= lon < grid_size

    idx = lat * grid_size + lon

    assert 0 <= idx < grid_size * grid_size

    return one_hot_encode(torch.tensor(idx), grid_size * grid_size)


def out_of_onehot(onehot, min_lat, max_lat, min_lon, max_lon, grid_size):
    onehot = torch.argmax(onehot)
    lat = onehot // grid_size
    lon = onehot % grid_size

    lat = denormalize_point(lat/grid_size, min_lat, max_lat)
    lon = denormalize_point(lon/grid_size, min_lon, max_lon)
    return (lat, lon)


def one_hot_encode(label, num_classes):
    return torch.nn.functional.one_hot(label.to(torch.int64), num_classes=num_classes)

# test dencode/encode
min_lat = -2
max_lat = 7
min_lon = 3
max_lon = 10
lat = 3.4
lon = 4.4
grid_size = 10
same = (out_of_onehot(into_onehot(lat, lon, min_lat, max_lat, min_lon, max_lon, grid_size), min_lat, max_lat, min_lon, max_lon, grid_size))
assert abs(same[0] - lat) < 0.01
assert abs(same[1] - lon) < 0.01

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

        self.grid_size = 5

        self.image_one_hot_labels_freq = {}
        for (_, [lat, lon]) in self.labels.items():
            idx = torch.argmax(
                self.real_location_to_label(lat, lon)
            )
            idx = int(idx)
            if idx in self.image_one_hot_labels_freq:
                self.image_one_hot_labels_freq[idx] += 1
            else:
                self.image_one_hot_labels_freq[idx] = 1

    def label_to_real_location(self, label):
        # lat, lon = label
        # return denormalize_point(lat, self.min_lat, self.max_lat), denormalize_point(lon, self.min_lon, self.max_lon)
        return out_of_onehot(label, self.min_lat, self.max_lat, self.min_lon, self.max_lon, self.grid_size)
    
    def real_location_to_label(self, lat, lon):
        return into_onehot(lat, lon, self.min_lat, self.max_lat, self.min_lon, self.max_lon, self.grid_size)

    def __len__(self):
        return len(self.labels.keys())

    def __getitem__(self, idx):
        image_name = list(self.labels.keys())[idx]
        image = Image.open(f'{self.root_dir}/{image_name}')

        if self.transform:
            image = self.transform(image)

        label = self.labels[image_name]
        [lat, lon] = label
        one_hot = into_onehot(lat, lon, self.min_lat, self.max_lat,
                              self.min_lon, self.max_lon, self.grid_size)
        label = torch.from_numpy(
            np.array(one_hot, dtype=np.float32)
        )
        sample = [image, label]

        return sample


if __name__ == "__main__":
    dataset = CampusImagesDataSet('processed_data')
    for i in range(4):
        sample = dataset[i]
        # print(i, sample['image'].shape, sample['label'])
        plt.imshow(sample['image'].permute(1, 2, 0))
        plt.show()
