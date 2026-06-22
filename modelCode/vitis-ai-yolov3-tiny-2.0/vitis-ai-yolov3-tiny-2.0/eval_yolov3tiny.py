# python eval_yolov3tiny --model_path logs/yolov3_tiny/epXXX-lossXX-val_lossXX.h5 --anchors_path configs/tiny_yolo3_anchors.txt  --classes_path configs/lane_class.txt  --annotation_file data/lane_detection/train/_annotations.txt  --quantize --eval_quant

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv3-Tiny Vitis-AI Quantization & Evaluation Script
 - 지원 기능:
   * Pascal VOC 스타일 mAP 계산 (한 개 IoU: 예, 0.5)
   * Float YOLOv3-Tiny 모델 평가
   * Vitis-AI PTQ (Post-Training Quantization)
   * Quantized YOLOv3-Tiny 모델 평가
"""

import os, argparse, time, sys
import numpy as np
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import operator

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow_model_optimization.quantization.keras import vitis_quantize

from yolo3.postprocess_np import yolo3_postprocess_np
from common.data_utils import preprocess_image
from common.utils import (
    get_dataset, get_classes, get_anchors, get_colors,
    draw_boxes, optimize_tf_gpu, get_custom_objects
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
optimize_tf_gpu(tf, K)

# ---------------------------------------------------
# 1. Evaluation Metrics & Helper Functions
# ---------------------------------------------------
def annotation_parse(annotation_lines, class_names):
    """
    annotation txt:
      image_path x1,y1,x2,y2,class_idx x1,y1,x2,y2,class_idx ...
    """
    annotation_records = OrderedDict()
    classes_records = OrderedDict({c: [] for c in class_names})
    for line in annotation_lines:
        line = line.strip()
        if len(line) == 0:
            continue
        parts = line.split(' ')
        image_name = parts[0]
        boxes = parts[1:]
        box_records = {}
        for box in boxes:
            if ',' not in box:
                continue
            # last field: class index
            cls_idx = int(box.split(',')[-1])
            class_name = class_names[cls_idx]
            coordinate = ','.join(box.split(',')[:-1])
            box_records[coordinate] = class_name
            record = [os.path.basename(image_name), coordinate]
            classes_records.setdefault(class_name, []).append(record)
        annotation_records[image_name] = box_records
    return annotation_records, classes_records


def box_iou(pred_box, gt_box):
    inter_x1 = max(pred_box[0], gt_box[0])
    inter_y1 = max(pred_box[1], gt_box[1])
    inter_x2 = min(pred_box[2], gt_box[2])
    inter_y2 = min(pred_box[3], gt_box[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    pred_area = max(0.0, (pred_box[2] - pred_box[0])) * max(0.0, (pred_box[3] - pred_box[1]))
    gt_area   = max(0.0, (gt_box[2] - gt_box[0]))   * max(0.0, (gt_box[3] - gt_box[1]))
    union = pred_area + gt_area - inter
    return 0.0 if union <= 0 else inter / union


def match_gt_box(pred_record, gt_records, iou_threshold=0.5):
    """
    pred_record: [image_file, 'xmin,ymin,xmax,ymax', score]
    gt_records:  [ [image_file,'xmin,ymin,xmax,ymax','unused'], ... ]
    """
    max_iou, max_index = 0.0, -1
    pred_box = [float(x) for x in pred_record[1].split(',')]
    for i, gt_record in enumerate(gt_records):
        gt_box = [float(x) for x in gt_record[1].split(',')]
        iou = box_iou(pred_box, gt_box)
        if iou > max_iou and gt_record[2] == 'unused' and pred_record[0] == gt_record[0]:
            max_iou, max_index = iou, i
    if max_iou < iou_threshold:
        max_index = -1
    return max_index


def voc_ap(rec, prec):
    rec = [0.0] + rec + [1.0]
    prec = [0.0] + prec + [0.0]
    mrec = rec[:]
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = [i for i in range(1, len(mrec)) if mrec[i] != mrec[i - 1]]
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def get_rec_prec(true_positive, false_positive, gt_records):
    cumsum = 0
    for i, v in enumerate(false_positive):
        false_positive[i] += cumsum
        cumsum += v
    cumsum = 0
    for i, v in enumerate(true_positive):
        true_positive[i] += cumsum
        cumsum += v

    rec = [tp / len(gt_records) if len(gt_records) != 0 else 0 for tp in true_positive]
    prec = [tp / (fp + tp) if (fp + tp) > 0 else 0 for fp, tp in zip(false_positive, true_positive)]
    return rec, prec


def calc_AP(gt_records, pred_records, class_name, iou_threshold):
    """
    gt_records : [ [image_file,'xmin,ymin,xmax,ymax'], ... ]
    pred_records : [ [image_file,'xmin,ymin,xmax,ymax',score], ... ] (score desc)
    """
    gt_records = [g + ['unused'] for g in gt_records]
    nd = len(pred_records)
    true_positive = [0] * nd
    false_positive = [0] * nd

    for idx, pred_record in enumerate(pred_records):
        image_gt = [g for g in gt_records if g[0] == pred_record[0]]
        i = match_gt_box(pred_record, image_gt, iou_threshold=iou_threshold)
        if i != -1:
            image_gt[i][2] = 'used'
            true_positive[idx] = 1
        else:
            false_positive[idx] = 1

    rec, prec = get_rec_prec(true_positive, false_positive, gt_records)
    ap, _, _ = voc_ap(rec, prec)
    return ap


def compute_mAP(annotation_records, gt_classes_records,
                pred_classes_records, class_names, iou_threshold):
    APs = {}
    for cname in class_names:
        if cname not in gt_classes_records:
            APs[cname] = 0.
            continue
        gt_records = gt_classes_records[cname]
        if cname not in pred_classes_records:
            APs[cname] = 0.
            continue
        pred_records = pred_classes_records[cname]
        ap = calc_AP(gt_records, pred_records, cname, iou_threshold)
        APs[cname] = ap

    mAP = np.mean(list(APs.values())) if len(APs) > 0 else 0.0
    return mAP, APs

# ---------------------------------------------------
# 2. YOLOv3-Tiny Prediction Logic
# ---------------------------------------------------
def predict_single_image(model, image, anchors, num_classes,
                         model_input_shape, conf_threshold,
                         nms_iou_threshold=0.5, elim_grid_sense=False):
    """
    YOLOv3 / YOLOv3-Tiny postprocess (yolo3_postprocess_np 사용)
    """
    image_data = preprocess_image(image, model_input_shape)
    image_shape = image.size[::-1]  # (h, w)

    prediction = model.predict([image_data])
    if not isinstance(prediction, list):
        prediction = [prediction]

    pred_boxes, pred_classes, pred_scores = yolo3_postprocess_np(
        prediction,
        image_shape,
        anchors,
        num_classes,
        model_input_shape,
        max_boxes=100,
        iou_threshold=nms_iou_threshold,
        confidence=conf_threshold,
        elim_grid_sense=elim_grid_sense
    )
    return pred_boxes, pred_classes, pred_scores


def get_prediction_records(model, annotation_records, anchors,
                           class_names, model_input_shape,
                           conf_threshold, nms_iou_threshold=0.5,
                           elim_grid_sense=False):
    """
    annotation_records : { image_path : { 'x1,y1,x2,y2':'classname', ... }, ... }
    return:
      pred_classes_records: {
        'class_name' : [ [img_name, 'x1,y1,x2,y2', score], ... (score desc) ]
      }
    """
    pred_classes_records = OrderedDict()
    pbar = tqdm(total=len(annotation_records), desc='Evaluating')

    for image_name in annotation_records.keys():
        image = Image.open(image_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        boxes, classes, scores = predict_single_image(
            model, image, anchors, len(class_names),
            model_input_shape, conf_threshold,
            nms_iou_threshold=nms_iou_threshold,
            elim_grid_sense=elim_grid_sense
        )

        if boxes is not None:
            for box, cls, score in zip(boxes, classes, scores):
                cname = class_names[cls]
                coord = "{},{},{},{}".format(*box)
                pred_classes_records.setdefault(cname, []).append(
                    [os.path.basename(image_name), coord, float(score)]
                )

        image.close()
        pbar.update(1)
    pbar.close()

    # score 내림차순 정렬
    for plist in pred_classes_records.values():
        plist.sort(key=lambda e: e[2], reverse=True)

    return pred_classes_records

# ---------------------------------------------------
# 3. Model Loading (YOLOv3-Tiny Inference)
# ---------------------------------------------------
def load_inference_model(model_path):
    """
    YOLOv3-Tiny Keras .h5 모델 로드
      - 학습 시 custom_objects를 사용한 모델이라고 가정
      - (keras-yolo3 계열, 혹은 직접 정의한 custom layer 포함)
    """
    print(f"[Info] Loading YOLOv3-Tiny Keras model: {model_path}")
    custom_objects = get_custom_objects()
    model = load_model(model_path, compile=False, custom_objects=custom_objects)
    model.summary()
    return model


def load_quantized_model(model_path):
    print(f"[Info] Loading Quantized Model: {model_path}")
    custom_objects = get_custom_objects()
    with vitis_quantize.quantize_scope():
        model = load_model(model_path, compile=False, custom_objects=custom_objects)
    model.summary()
    return model

# ---------------------------------------------------
# 4. Main Execution (PTQ + Eval)
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='YOLOv3-Tiny Vitis-AI Quantization & Evaluation'
    )

    # Path Arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to Float YOLOv3-Tiny .h5 model')
    parser.add_argument('--anchors_path', type=str,
                        default='configs/tiny_yolo3_anchors.txt',
                        help='Path to YOLOv3-Tiny anchors txt')
    parser.add_argument('--classes_path', type=str,
                        default='configs/lane_class.txt',
                        help='Path to class names txt')
    parser.add_argument('--annotation_file', type=str,
                        default='data/lane_detection/train/_annotations.txt',
                        help='Path to test annotation txt')
    parser.add_argument('--output_dir', type=str,
                        default='./quantized_yolov3_tiny',
                        help='Directory to save quantized model & results')

    # Operation Flags
    parser.add_argument('--eval_float', action='store_true',
                        help='Evaluate Float YOLOv3-Tiny model')
    parser.add_argument('--quantize', action='store_true',
                        help='Run Vitis-AI PTQ for YOLOv3-Tiny')
    parser.add_argument('--eval_quant', action='store_true',
                        help='Evaluate Quantized YOLOv3-Tiny model')

    # Parameters
    parser.add_argument('--model_input_shape', type=str,
                        default='256x256',
                        help='Model input shape, e.g. 416x416')
    parser.add_argument('--iou_threshold', type=float,
                        default=0.5,
                        help='IoU threshold for mAP')
    parser.add_argument('--conf_threshold', type=float,
                        default=0.001,
                        help='Confidence threshold for detection')
    parser.add_argument('--nms_iou_threshold', type=float,
                        default=0.5,
                        help='IoU threshold for NMS')
    parser.add_argument('--elim_grid_sense', action='store_true',
                        help='Eliminate grid sensitivity (yolo3_postprocess_np 옵션)')
    parser.add_argument('--quant_batch_size', type=int,
                        default=16,
                        help='Calibration batch size')
    parser.add_argument('--quant_steps', type=int,
                        default=100,
                        help='Calibration steps')

    args = parser.parse_args()

    # 기본 설정
    anchors = get_anchors(args.anchors_path)
    class_names = get_classes(args.classes_path)
    h, w = args.model_input_shape.split('x')
    input_shape = (int(h), int(w))

    os.makedirs(args.output_dir, exist_ok=True)
    quantized_model_path = os.path.join(args.output_dir, 'quantized_yolov3_tiny.h5')

    # 데이터셋 로드
    lines = get_dataset(args.annotation_file, shuffle=False)
    annotation_records, gt_classes_records = annotation_parse(lines, class_names)

    # 1) Float 모델 평가
    if args.eval_float:
        print("\n=== [Task] Evaluate Float YOLOv3-Tiny Model ===")
        float_model = load_inference_model(args.model_path)

        pred_records = get_prediction_records(
            float_model,
            annotation_records,
            anchors,
            class_names,
            input_shape,
            args.conf_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            elim_grid_sense=args.elim_grid_sense
        )
        mAP, APs = compute_mAP(
            annotation_records,
            gt_classes_records,
            pred_records,
            class_names,
            args.iou_threshold
        )

        print(f"\n[Float Model Result] mAP@{args.iou_threshold}: {mAP*100:.2f}%")
        for cls, ap in APs.items():
            print(f" - {cls}: {ap*100:.2f}%")

    # 2) Quantization (PTQ)
    if args.quantize:
        print("\n=== [Task] Vitis-AI Quantization (PTQ) for YOLOv3-Tiny ===")
        float_model = load_inference_model(args.model_path)

        # Calibration dataset generator
        def calib_dataset_gen():
            # steps * batch_size 만큼만 사용
            max_imgs = args.quant_batch_size * args.quant_steps
            image_list = list(annotation_records.keys())[:max_imgs]
            for image_name in image_list:
                img = Image.open(image_name)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                yield [preprocess_image(img, input_shape)]

        # VitisQuantizer 실행
        quantizer = vitis_quantize.VitisQuantizer(float_model)
        print("[Info] Starting Quantization...")
        q_model = quantizer.quantize_model(
            calib_dataset=calib_dataset_gen(),
            calib_batch_size=args.quant_batch_size,
            calib_steps=args.quant_steps,
            verbose=1
        )

        q_model.save(quantized_model_path)
        print(f"[Info] Quantized YOLOv3-Tiny model saved to: {quantized_model_path}")

    # 3) Quantized 모델 평가
    if args.eval_quant:
        print("\n=== [Task] Evaluate Quantized YOLOv3-Tiny Model ===")
        if not os.path.exists(quantized_model_path):
            print(f"[Error] Quantized model not found at {quantized_model_path}. Run --quantize first.")
            return

        q_model = load_quantized_model(quantized_model_path)

        pred_records = get_prediction_records(
            q_model,
            annotation_records,
            anchors,
            class_names,
            input_shape,
            args.conf_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            elim_grid_sense=args.elim_grid_sense
        )
        mAP, APs = compute_mAP(
            annotation_records,
            gt_classes_records,
            pred_records,
            class_names,
            args.iou_threshold
        )

        print(f"\n[Quantized Model Result] mAP@{args.iou_threshold}: {mAP*100:.2f}%")
        for cls, ap in APs.items():
            print(f" - {cls}: {ap*100:.2f}%")

if __name__ == '__main__':
    main()
