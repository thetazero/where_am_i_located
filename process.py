from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os
import json


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

def rescale_image(image_path, output_path):
    image = Image.open(image_path)
    resized_image = image.resize((100, 100))
    resized_image.save(output_path)

if __name__ == "__main__":
    image_path = "path_to_your_image.png"
    raw_data_path = "raw_data"
    output_path = "processed_data"

    mkdir(output_path)

    labels = {}

    for file in os.listdir(raw_data_path):
        image_path = os.path.join(raw_data_path, file)
        labels[file] = process(image_path)
        rescale_image(image_path, os.path.join(output_path, file))

    json.dump(labels, open(os.path.join(output_path, "labels.json",), "w"), indent=4)
    
