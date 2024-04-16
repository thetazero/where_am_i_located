from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import json
import os
from tqdm import tqdm

import plotly.express as px
import pandas as pd


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
    try:
        image = Image.open(image_path)
        exif = image._getexif()

        return get_lat_lon(exif)
    except Exception as e:
        print(f"Error processing {image_path}")
        raise e


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


def plot_coordinates(coordinates):
    boundaries = [
        (40.444512, -79.948693, "Upper Left", "Boundary"),
        (40.444512, -79.936835, "Upper Right", "Boundary"),
        (40.440023, -79.937764, "Bottom Right", "Boundary"),
        (40.440023, -79.948693, "Bottom Left", "Boundary")
    ]

    coords = [
        (lat, lon, name, "Raw") for lat, lon, name in coordinates
    ]

    df = pd.DataFrame(boundaries + coords, columns=[

        'lat', 'lon', 'name', 'type'])
    fig = px.scatter_mapbox(df,
                            lat='lat',
                            lon='lon',
                            color='type',
                            size=[100 for _ in range(len(df))],
                            hover_data=['name'],
                            zoom=15,
                            height=800,
                            width=800)

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


if __name__ == "__main__":
    group = "train"  # train/test/val

    raw_data_path = f"data/{group}/raw"
    output_path = f"data/{group}/processed"

    stomp = False  # Set to True if you want to overwrite the existing output data

    mkdir(output_path)

    labels = {}

    total = len(os.listdir(raw_data_path))

    for file in tqdm(os.listdir(raw_data_path), total=total, desc="Processing images"):

        image_path = os.path.join(raw_data_path, file)
        labels[file] = process(image_path)
        if stomp or (not os.path.exists(os.path.join(output_path, file))):
            rescale_image(image_path, os.path.join(output_path, file))

    json.dump(labels, open(os.path.join(
        output_path, "labels.json",), "w"), indent=4)

    coordinates = [
        (value[0], value[1], key) for key, value in labels.items()
    ]

    plot_coordinates(coordinates)
