"""
Gintautas Plonis 1812957
EfficientDet | Focal loss | Raven, Coffee, Headphones
(Optional) REST API
"""
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import Resizer, Normalizer, Augmenter, collater, OpenImagesDataset
from src.model import EfficientDet
from tensorboardX import SummaryWriter
import shutil
import numpy as np
from tqdm.autonotebook import tqdm


def get_args():
    parser = argparse.ArgumentParser("EfficientDet: Scalable and Efficient Object Detection")
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
    parser.add_argument("--log_path", type=str, default="tensorboard/efficientdet")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--model_name", type=str, default="efficientdet")

    args = parser.parse_args()
    return args


def train(opt):
    if not os.path.isdir(opt.data_path):
        print(f"Data for dataset not found at {opt.data_path}")
        return

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

    test_set = OpenImagesDataset(root_dir=opt.data_path, set_name="val",
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

                progress_bar.set_description(f'Epoch: {epoch + 1}/{opt.num_epochs} | '
                                             f'Iteration: {iter + 1}/{num_iter_per_epoch} | '
                                             f'Cls loss: {cls_loss:.5f}. Reg loss: {reg_loss:.5f} | '
                                             f'Batch loss: {loss:.5f} Total loss: {total_loss:.5f}')

                writer.add_scalar('Train/Total_loss', total_loss, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Regression_loss', reg_loss, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Classification_loss (focal loss)', cls_loss, epoch * num_iter_per_epoch + iter)

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

            print(f'Epoch: {epoch + 1}/{opt.num_epochs} | '
                  f'Classification loss: {cls_loss:1.5f} | '
                  f'Regression loss: {reg_loss:1.5f} | Total loss: {np.mean(loss):1.5f}')

            writer.add_scalar('Test/Total_loss', loss, epoch)
            writer.add_scalar('Test/Regression_loss', reg_loss, epoch)
            writer.add_scalar('Test/Classification_loss (focal loss)', cls_loss, epoch)

            if loss + opt.es_min_delta < best_loss:
                best_loss = loss
                best_epoch = epoch
                torch.save(model, os.path.join(opt.saved_path, f'{opt.model_name}.pth'))

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print(f"Stop training at epoch {epoch}. The lowest loss achieved is {loss}")
                break

    torch.save(model, os.path.join(opt.saved_path, f'{opt.model_name}-final.pth'))
    writer.close()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
