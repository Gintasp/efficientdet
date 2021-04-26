import math
import os
import argparse

import glob2 as glob2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import OPEN_IMAGES_CLASSES
from src.dataset import Resizer, Normalizer, Augmenter, collater, OpenImagesDataset
from src.model import EfficientDet
from tensorboardX import SummaryWriter
import shutil
import numpy as np
from tqdm.autonotebook import tqdm
from openimages.download import download_dataset


def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
    parser.add_argument("--image_size", type=int, default=512, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=8, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--data_path", type=str, default="data", help="the root folder of dataset")
    parser.add_argument("--log_path", type=str, default="tensorboard/signatrix_efficientdet_coco")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--num_samples", type=int, default=100, help="number of training images to download")

    args = parser.parse_args()
    return args


def train(opt):
    num_gpus = 1
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    training_params = {"batch_size": opt.batch_size * num_gpus,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": collater,
                       "num_workers": 12}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": collater,
                   "num_workers": 12}

    training_set = OpenImagesDataset(root_dir=opt.data_path, set_name="train",
                                     transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    training_loader = DataLoader(training_set, **training_params)

    test_set = OpenImagesDataset(root_dir=opt.data_path, set_name="test",
                                 transform=transforms.Compose([Normalizer(), Resizer()]))
    test_loader = DataLoader(test_set, **test_params)

    model = EfficientDet(num_classes=training_set.num_classes())

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    writer = SummaryWriter(opt.log_path)
    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    best_loss = 1e5
    best_epoch = 0
    model.train()

    num_iter_per_epoch = len(training_loader)
    for epoch in range(opt.num_epochs):
        model.train()
        epoch_loss = []
        progress_bar = tqdm(training_loader)

        for iter, data in enumerate(progress_bar):
            try:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])
                else:
                    cls_loss, reg_loss = model([data['img'].float(), data['annot']])

                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()
                loss = cls_loss + reg_loss
                if loss == 0:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                epoch_loss.append(float(loss))
                total_loss = np.mean(epoch_loss)

                progress_bar.set_description(
                    'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Batch loss: {:.5f} Total loss: {:.5f}'.format(
                        epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss, reg_loss, loss,
                        total_loss))
                writer.add_scalar('Train/Total_loss', total_loss, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Regression_loss', reg_loss, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Classfication_loss (focal loss)', cls_loss, epoch * num_iter_per_epoch + iter)

            except Exception as e:
                print(e)
                continue
        scheduler.step(np.mean(epoch_loss))

        if epoch % opt.test_interval == 0:
            model.eval()
            loss_regression_ls = []
            loss_classification_ls = []
            for iter, data in enumerate(test_loader):
                with torch.no_grad():
                    if torch.cuda.is_available():
                        cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])
                    else:
                        cls_loss, reg_loss = model([data['img'].float(), data['annot']])

                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss_classification_ls.append(float(cls_loss))
                    loss_regression_ls.append(float(reg_loss))

            cls_loss = np.mean(loss_classification_ls)
            reg_loss = np.mean(loss_regression_ls)
            loss = cls_loss + reg_loss

            print(
                'Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                    epoch + 1, opt.num_epochs, cls_loss, reg_loss,
                    np.mean(loss)))
            writer.add_scalar('Test/Total_loss', loss, epoch)
            writer.add_scalar('Test/Regression_loss', reg_loss, epoch)
            writer.add_scalar('Test/Classfication_loss (focal loss)', cls_loss, epoch)

            if loss + opt.es_min_delta < best_loss:
                best_loss = loss
                best_epoch = epoch
                torch.save(model, os.path.join(opt.saved_path, "efficientdet.pth"))

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, loss))
                break

    torch.save(model, os.path.join(opt.saved_path, "efficientdet-final.pth"))
    writer.close()


def create_train_data(class_name):
    print(f"\nCreating train data for class \'{class_name}\'")
    img_files = glob2.glob(f"data/{class_name}/images/*jpg")
    print(f"Train images: {math.ceil(len(img_files) * .9)}")
    for f in tqdm(img_files[:math.ceil(len(img_files) * .9)]):  # Use 90% of images for training
        try:
            id = f[-20:-4]
            os.replace(f'data/{class_name}/images/{id}.jpg', f'data/train/{class_name}/images/{id}.jpg', )
            os.replace(f'data/{class_name}/pascal/{id}.xml', f'data/train/{class_name}/pascal/{id}.xml')
        except Exception as e:
            print(e)


def create_test_data(class_name):
    print(f"\nCreating test data for class \'{class_name}\'")
    img_files = glob2.glob(f"data/{class_name}/images/*jpg")
    print(f"Test images: {math.ceil(len(img_files) * .9)}")
    for f in tqdm(img_files[:math.ceil(len(img_files) * .9)]):  # Use remaining 10% of images for testing
        try:
            id = f[-20:-4]
            os.replace(f'data/{class_name}/images/{id}.jpg', f'data/test/{class_name}/images/{id}.jpg')
            os.replace(f'data/{class_name}/pascal/{id}.xml', f'data/test/{class_name}/pascal/{id}.xml')
        except Exception as e:
            print(e)


def setup_data(opt):
    data_dir = opt.data_path
    number_for_samples = opt.num_samples

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    else:
        print('Data already downloaded, skipping...')
        return

    print('Downloading data...')
    download_dataset(data_dir, [c.capitalize for c in OPEN_IMAGES_CLASSES],
                     limit=number_for_samples, annotation_format="pascal")

    print('Creating data folder structure...')
    for c in OPEN_IMAGES_CLASSES:
        os.makedirs(f'data/train/{c}/images', exist_ok=True)
        os.makedirs(f'data/test/{c}/images', exist_ok=True)
        os.makedirs(f'data/train/{c}/pascal', exist_ok=True)
        os.makedirs(f'data/test/{c}/pascal', exist_ok=True)

    for c in OPEN_IMAGES_CLASSES:
        try:
            create_train_data(c)
            create_test_data(c)
            shutil.rmtree(f'data/{c}')
        except Exception as e:
            print(e)

    print('Done!')


if __name__ == "__main__":
    opt = get_args()
    setup_data(opt)
    train(opt)
