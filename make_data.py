"""
Gintautas Plonis 1812957
EfficientDet | Focal loss | Raven, Coffee, Headphones
(Optional) REST API
"""
import math
import os
import shutil
import argparse

import glob2
from openimages.download import download_dataset
from tqdm import tqdm

from src.config import OPEN_IMAGES_CLASSES


def get_args():
    parser = argparse.ArgumentParser("EfficientDet: Scalable and Efficient Object Detection")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of total images to download")
    parser.add_argument("--data_path", type=str, default="data", help="The root folder of dataset")
    args = parser.parse_args()

    return args


def create_train_data(opt, class_name):
    print(f"\nCreating train data for class \'{class_name}\'")
    img_files = glob2.glob(f"{opt.data_path}/{class_name}/images/*jpg")
    print(f"Train images: {math.ceil(len(img_files) * .9)}")
    for f in tqdm(img_files[:math.ceil(len(img_files) * .9)]):  # Use 90% of images for training
        try:
            id = f[-20:-4]
            os.replace(f'{opt.data_path}/{class_name}/images/{id}.jpg', f'{opt.data_path}/train/{class_name}/images/{id}.jpg', )
            os.replace(f'{opt.data_path}/{class_name}/pascal/{id}.xml', f'{opt.data_path}/train/{class_name}/pascal/{id}.xml')
        except Exception as e:
            print(e)


def create_val_data(opt, class_name):
    print(f"\nCreating validation data for class \'{class_name}\'")
    img_files = glob2.glob(f"{opt.data_path}/{class_name}/images/*jpg")
    print(f"Validation images: {math.ceil(len(img_files) * .9)}")
    for f in tqdm(img_files[:math.ceil(len(img_files) * .9)]):  # Use remaining 10% of images for testing
        try:
            id = f[-20:-4]
            os.replace(f'{opt.data_path}/{class_name}/images/{id}.jpg', f'{opt.data_path}/val/{class_name}/images/{id}.jpg')
            os.replace(f'{opt.data_path}/{class_name}/pascal/{id}.xml', f'{opt.data_path}/val/{class_name}/pascal/{id}.xml')
        except Exception as e:
            print(e)


def setup_data(opt):
    """
    Download training and validation data and store it in compatible folder structure
    :param opt: CLI options
    """
    data_dir = opt.data_path

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    else:
        print('Data already downloaded, skipping...')
        return

    print('Downloading data...')
    download_dataset(data_dir, [c.capitalize() for c in OPEN_IMAGES_CLASSES],
                     limit=opt.num_samples, annotation_format="pascal")

    print('Creating data folder structure...')
    for c in OPEN_IMAGES_CLASSES:
        os.makedirs(f'{data_dir}/train/{c}/images', exist_ok=True)
        os.makedirs(f'{data_dir}/val/{c}/images', exist_ok=True)
        os.makedirs(f'{data_dir}/train/{c}/pascal', exist_ok=True)
        os.makedirs(f'{data_dir}/val/{c}/pascal', exist_ok=True)

    for c in OPEN_IMAGES_CLASSES:
        try:
            create_train_data(opt, c)
            create_val_data(opt, c)
            shutil.rmtree(f'{data_dir}/{c}')
        except Exception as e:
            print(e)

    print('Done!\n==========================')
    print('Downloaded images:')
    for c in OPEN_IMAGES_CLASSES:
        train = glob2.glob(f'{opt.data_path}/train/{c}/images/*jpg')
        val = glob2.glob(f'{opt.data_path}/val/{c}/images/*jpg')
        print(f'{c.capitalize()}: {len(train)} training, {len(val)} validation')


if __name__ == "__main__":
    opt = get_args()
    setup_data(opt)
