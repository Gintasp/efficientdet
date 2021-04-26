import os
import argparse
import torch
from torchvision import transforms
from tqdm import tqdm

from src.dataset import Resizer, Normalizer, OpenImagesDataset
from src.config import OPEN_IMAGES_COLORS, OPEN_IMAGES_CLASSES
import cv2
import shutil


def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
    parser.add_argument("--image_size", type=int, default=512, help="The common width and height for all images")
    parser.add_argument("--data_path", type=str, default="data", help="The root folder of dataset")
    parser.add_argument("--cls_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--pretrained_model", type=str, default="trained_models/efficientdet.pth")
    parser.add_argument("--output", type=str, default="predictions")
    args = parser.parse_args()

    return args


def test(opt):
    if torch.cuda.is_available():
        model = torch.load(opt.pretrained_model).module.cuda()
    else:
        model = torch.load(opt.pretrained_model, map_location=torch.device('cpu')).module

    dataset = OpenImagesDataset(root_dir=opt.data_path, set_name='val',
                                transform=transforms.Compose([Normalizer(), Resizer()]))

    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)

    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]
        scale = data['scale']
        with torch.no_grad():
            if torch.cuda.is_available():
                scores, labels, boxes = model(data['img'].cuda().permute(2, 0, 1).float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            boxes /= scale

        if boxes.shape[0] > 0:
            class_name = dataset.image_to_category_name[dataset.images[idx]]
            path = f'{opt.data_path}/val/{class_name}/images/{dataset.images[idx]}.jpg'
            output_image = cv2.imread(path)

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

            cv2.imwrite(f"{opt.output}/{dataset.images[idx]}_prediction.jpg", output_image)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
