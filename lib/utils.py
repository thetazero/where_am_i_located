import matplotlib.pyplot as plt
import numpy as np


def plot_torch_image(img_tensor):
    plt.figure()
    img_tensor = img_tensor.permute(1, 2, 0)
    plt.imshow(img_tensor)

    plt.show()


def plot_image_from_disk(image_path):
    plt.figure()
    plt.imshow(plt.imread(image_path))

    plt.show()


def plot_one_hot_vectors(vectors, labels):
    num_vectors = len(vectors)
    num_categories = len(vectors[0])
    categories = np.arange(num_categories)

    width = 0.8 / num_vectors  # Adjust the width of bars based on the number of vectors
    plt.figure()

    for i, vector in enumerate(vectors):
        plt.bar(categories + i * width, vector, width=width,
                align='center', label=labels[i])

    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title('One-Hot Vectors Representation')
    plt.xticks(categories + 0.4, categories)
    plt.yticks([0, 1])
    plt.legend()
    plt.show()
