import os
import argparse

import numpy as np
import torch
from torchvision import transforms

from src.dataset import Resizer, Normalizer
from src.config import OPEN_IMAGES_COLORS, OPEN_IMAGES_CLASSES
import cv2
import ntpath
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
    parser.add_argument("--image_path", type=str, default=None, help="Image path to perform localization on")
    parser.add_argument("--cls_threshold", type=float, default=0.5)
    parser.add_argument("--pretrained_model", type=str, default="trained_models/efficientdet-final.pth")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    return args


def get_image_data(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.

    sample = {'img': img, 'annot': np.zeros((0, 5))}

    transform = transforms.Compose([Normalizer(), Resizer()])
    sample = transform(sample)

    return sample


def demo(opt):
    if torch.cuda.is_available():
        model = torch.load(opt.pretrained_model).module.cuda()
    else:
        model = torch.load(opt.pretrained_model, map_location=torch.device('cpu')).module

    data = get_image_data(opt.image_path)
    scale = data['scale']
    with torch.no_grad():
        if torch.cuda.is_available():
            scores, labels, boxes = model(data['img'].cuda().permute(2, 0, 1).float().unsqueeze(dim=0))
        else:
            scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
        boxes /= scale

    if boxes.shape[0] > 0:
        output_image = cv2.imread(opt.image_path)

        for box_id in range(boxes.shape[0]):
            pred_prob = float(scores[box_id])
            if pred_prob < opt.cls_threshold:
                break
            pred_label = int(labels[box_id])
            x1, y1, x2, y2 = boxes[box_id, :]
            color = OPEN_IMAGES_COLORS[pred_label]
            start_point = (int(x1), int(y1))
            end_point = (int(x2), int(y2))

            cv2.rectangle(output_image, start_point, end_point, color, thickness=2)
            text_size = cv2.getTextSize(f'{OPEN_IMAGES_CLASSES[pred_label]}: {pred_prob:.2f}',
                                        cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

            cv2.rectangle(output_image, start_point,
                          (int(x1 + text_size[0] + 3), int(y1 + text_size[1] + 4)), color, thickness=-1)
            cv2.putText(
                output_image, f'{OPEN_IMAGES_CLASSES[pred_label]}: {pred_prob:.2f}',
                (int(x1), int(y1 + text_size[1] + 4)), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 255), 1)

        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        if opt.output_dir is not None:
            if not os.path.isdir(opt.output_dir):
                os.makedirs(opt.output_dir)
            out_filename = os.path.splitext(ntpath.basename(opt.image_path))[0]
            cv2.imwrite(f"{opt.output_dir}/{out_filename}_prediction.jpg", output_image)


if __name__ == "__main__":
    opt = get_args()
    demo(opt)
