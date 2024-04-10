from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import json
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def get_geotagging(exif):
    if not exif:
        raise ValueError("No EXIF metadata found")

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")

            for (t, value) in GPSTAGS.items():
                if t in exif[idx]:
                    geotagging[value] = exif[idx][t]

    return geotagging


def get_decimal_from_dms(dms, ref):
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return round(degrees + minutes + seconds, 5)


def get_lat_lon(exif):
    geotagging = get_geotagging(exif)
    lat = get_decimal_from_dms(
        geotagging['GPSLatitude'], geotagging['GPSLatitudeRef'])
    lon = get_decimal_from_dms(
        geotagging['GPSLongitude'], geotagging['GPSLongitudeRef'])

    return (lat, lon)


def process(image_path):
    image = Image.open(image_path)
    exif = image._getexif()

    return get_lat_lon(exif)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rescale_image(image_path, output_path, new_size=(200, 200)):
    image = Image.open(image_path)
    resized_image = image.resize(new_size)
    resized_image.save(output_path)

def make_ext_from_coords(a, b):
    ax, ay = a
    bx, by = b
    return (min(ax, bx), max(ax, bx), min(ay, by), max(ay, by))

def plot_coordinates(coordinates, output_file):
    upper_left = (40.444512, -79.948693)
    bottom_right = (40.440023, -79.937764)

    landmarks = [
        (
            (40.444162, -79.942764),
            "Walking to the sky"),
        (
            (40.443274, -79.940583),
            "Gesling Stadium"
        ),
        (
            (40.443880, -79.944273),
            "Gates Hillman Center"
        ),
        (
            (40.441877, -79.947075),
            "Scaife Hall"
        ),
        (
            (40.441252, -79.943507),
            "Hunt Library"
        ),
        (
            (40.442207, -79.943300),
            "The Fence"
        ),
        (
            (40.441995, -79.940499),
            "Donner House"
        ),
        (
            (40.441673, -79.946022),
            "Porter Hall"
        )
    ]

    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]

    plt.figure(figsize=(10, 10))
    plt.scatter(x, y)

    plt.scatter(upper_left[0], upper_left[1], color='red', s=200)
    plt.scatter(bottom_right[0], bottom_right[1], color='red', s=200)

    for (location, name) in landmarks:
        plt.scatter(location[0], location[1], color='green', s=200)
        plt.text(location[0], location[1], name, color='black', fontsize=12)

    # background_img = Image.open("bg.png")

    # ext = make_ext_from_coords(upper_left, bottom_right)
    # plt.imshow(
    #     background_img,
    #     extent=ext
    # )
    # print(background_img.size)
    # aspect=background_img.size[0]/float(background_img.size[1])*((ext[1]-ext[0])/(ext[3]-ext[2]))
    # print(aspect)
    # plt.gca().set_aspect(aspect/3)


    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Coordinate Plot')

    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0, right=1, top=1, wspace=0, hspace=0)


    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    image_path = "path_to_your_image.png"
    raw_data_path = "raw_data"
    output_path = "processed_data"

    stomp = False  # Set to True if you want to overwrite the existing output data

    mkdir(output_path)

    labels = {}

    total = len(os.listdir(raw_data_path))

    for file in tqdm(os.listdir(raw_data_path), total=total, desc="Processing images"):

        image_path = os.path.join(raw_data_path, file)
        labels[file] = process(image_path)
        if stomp or not os.path.exists(os.path.join(output_path, file)):
            rescale_image(image_path, os.path.join(output_path, file))

    json.dump(labels, open(os.path.join(
        output_path, "labels.json",), "w"), indent=4)

    plot_coordinates(labels.values(), "picture_map.png")
