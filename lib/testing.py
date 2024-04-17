import torch
import numpy as np
from torch.utils.data import DataLoader

from . import load_model
from . import data_loader


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


def from_disk_eval(model_location, grid_size, data_loc, idx):
    model = load_model(model_location, grid_size)

    dl = DataLoader(
        data_loader.CampusImagesDataSet(
            data_loc, transforms=data_loader.all_transforms, grid_size=grid_size),
        shuffle=False,
    )

    pic = list(dl)[idx]
    plot_torch_image(pic[0][0])
    pred_loc = project_lib.eval(model, pic[0])
    pred_arr = pred_loc.cpu().numpy()[0]
    true_arr = pic[1].numpy()[0]
    plot_one_hot_vectors([pred_arr, true_arr], [
                         "predicted location", "true location"])
    (f"data/test/raw/{list(test_data.labels.keys())[idx]}", pred_loc)
