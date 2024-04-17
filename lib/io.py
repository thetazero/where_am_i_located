from . import CMUGuesserNet
import torch

def load_model(path, grid_size):
    model = CMUGuesserNet(grid_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)
