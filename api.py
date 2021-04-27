"""
Gintautas Plonis 1812957
EfficientDet | Focal loss | Raven, Coffee, Headphones
(Optional) REST API
"""
import ntpath
import os

import cv2
import flask
import torch
from flask import json, request, send_file
from werkzeug.utils import secure_filename

from demo import get_image_data
from src.config import OPEN_IMAGES_COLORS, OPEN_IMAGES_CLASSES

app = flask.Flask(__name__)
app.config["DEBUG"] = True
UPLOADS_PATH = 'web/uploads'


def predict(filename, thresh=0.5, model='trained_models/efficientdet-final.pth'):
    if torch.cuda.is_available():
        model = torch.load(model).module.cuda()
    else:
        model = torch.load(model, map_location=torch.device('cpu')).module

    path = f'{UPLOADS_PATH}/{filename}'
    data = get_image_data(path)
    scale = data['scale']
    with torch.no_grad():
        if torch.cuda.is_available():
            scores, labels, boxes = model(data['img'].cuda().permute(2, 0, 1).float().unsqueeze(dim=0))
        else:
            scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
        boxes /= scale

    if boxes.shape[0] > 0:
        output_image = cv2.imread(path)

        for box_id in range(boxes.shape[0]):
            pred_prob = float(scores[box_id])
            if pred_prob < thresh:
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

        out_filename = os.path.splitext(ntpath.basename(path))[0]
        cv2.imwrite(f"{UPLOADS_PATH}/{out_filename}_prediction.jpg", output_image)


@app.route('/', methods=['GET'])
def index():
    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/predict', methods=['POST'])
def post():
    """
    Takes 'file' form field with jpg image and returns prediction
    """
    if 'file' not in request.files:
        return json.dumps({'success': False}), 400, {'ContentType': 'application/json'}

    file = request.files['file']
    filename = secure_filename(file.filename)
    path = f'web/uploads/{filename}'
    file.save(path)

    predict(filename)
    out_filename = os.path.splitext(ntpath.basename(path))[0]

    return send_file(f'{UPLOADS_PATH}/{out_filename}_prediction.jpg')


app.run()
