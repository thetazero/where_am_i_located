from .network import CMUGuesserNet
from .io import load_model, save_model
from .testing import eval, from_disk_eval
from .data_loader import CampusImagesDataSet
from .autoencoder import MobileNetV2CAE


def model_string(grid_size=2, epochs=20):
    return f'model-gridsize{grid_size}-epochs{epochs}.torch'
