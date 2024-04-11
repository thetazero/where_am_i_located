import matplotlib.pyplot as plt

def plot_torch_image(img_tensor):
    img_tensor = img_tensor.permute(1, 2, 0)
    plt.imshow(img_tensor)
