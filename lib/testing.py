import torch
import numpy as np
import os
from torch.utils.data import DataLoader

from . import load_model
from . import data_loader
from .utils import plot_torch_image, plot_one_hot_vectors, plot_image_from_disk


def eval(model, image):
    """
    TODO: Split this into two things
    - Eval processed predict from pre-processed image to normalized lat+lon
    - Eval raw, takes a raw image procssses it, sends it to one above, then extracts real location
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        x = torch.from_numpy(np.array([image.numpy()]))
        x = image.to(device)

        yhat = model(x)

    return yhat


def from_disk_eval(model_location, grid_size=2, data_source="train", idx=0):
    model = load_model(model_location, grid_size)
    return nice_eval(model, grid_size, data_source, idx)


def nice_eval(model, grid_size=2, data_source="train", idx=0):
    proc_images_dataloader = DataLoader(
        data_loader.CampusImagesDataSet(
            f"data/{data_source}/processed", transform=data_loader.all_transforms, grid_size=grid_size),
        shuffle=False,
    )

    # show original image
    image_raw_name = os.listdir(f"data/{data_source}/raw")[idx]
    plot_image_from_disk(f"data/{data_source}/raw/{image_raw_name}")

    # show processed image
    pic = list(proc_images_dataloader)[idx]
    plot_torch_image(pic[0][0])

    pred_loc = eval(model, pic[0])
    pred_arr = pred_loc.cpu().numpy()[0]
    true_arr = pic[1].numpy()[0]
    plot_one_hot_vectors([pred_arr, true_arr], [
                         "predicted location", "true location"])  # show predicted vs true location
    return pred_loc
