#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""custom model callbacks."""
import os, sys, random, tempfile
import glob
import numpy as np

from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.callbacks import Callback

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from yolo3.model import get_yolo3_model


############################################################
# YOLOv3/Tiny-YOLOv3 전용 eval_AP 모듈
############################################################

from collections import OrderedDict
from PIL import Image
from tqdm import tqdm

from common.data_utils import preprocess_image
from common.utils import draw_boxes, get_colors
from yolo3.postprocess_np import yolo3_postprocess_np


class DatasetShuffleCallBack(Callback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        np.random.shuffle(self.dataset)


class CheckpointCleanCallBack(Callback):
    def __init__(self, checkpoint_dir, max_val_keep=5, max_eval_keep=2):
        self.checkpoint_dir = checkpoint_dir
        self.max_val_keep = max_val_keep
        self.max_eval_keep = max_eval_keep

    def on_epoch_end(self, epoch, logs=None):
        # 전체 체크포인트
        all_ckpts = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, 'ep*.h5')),
            reverse=False
        )
        # mAP 포함된 eval ckpt / 그 외 val ckpt 분리
        eval_ckpts = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, 'ep*-mAP*.h5')),
            reverse=False
        )
        val_ckpts = sorted(list(set(all_ckpts) - set(eval_ckpts)), reverse=False)

        # 최근 max_val_keep 개만 남기고 삭제
        for ckpt in val_ckpts[:-(self.max_val_keep)]:
            os.remove(ckpt)

        # 최근 max_eval_keep 개만 남기고 삭제
        for ckpt in eval_ckpts[:-(self.max_eval_keep)]:
            os.remove(ckpt)


class EvalCallBack(Callback):
    def __init__(self,
                 model_type,
                 annotation_lines,
                 anchors,
                 class_names,
                 model_input_shape,
                 model_pruning,
                 log_dir,
                 eval_epoch_interval=10,
                 save_eval_checkpoint=False,
                 elim_grid_sense=False):
        self.model_type = model_type
        self.annotation_lines = annotation_lines
        self.anchors = anchors
        self.class_names = class_names
        self.model_input_shape = model_input_shape
        self.model_pruning = model_pruning
        self.log_dir = log_dir
        self.eval_epoch_interval = eval_epoch_interval
        self.save_eval_checkpoint = save_eval_checkpoint
        self.elim_grid_sense = elim_grid_sense
        self.best_mAP = 0.0

        self.eval_model = self.get_eval_model()

    def get_eval_model(self):
        """
        YOLOv3 / Tiny-YOLOv3 전용 평가용 모델 생성
        """
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        num_feature_layers = num_anchors // 3  # 9→3, 6→2

        if self.model_type.startswith('yolo3_') or \
           self.model_type.startswith('yolo4_') or \
           self.model_type.startswith('tiny_yolo3_') or \
           self.model_type.startswith('tiny_yolo4_'):
            # YOLOv3 / YOLOv4 / Tiny 계열
            eval_model, _ = get_yolo3_model(
                self.model_type,
                num_feature_layers,
                num_anchors,
                num_classes,
                input_shape=self.model_input_shape + (3,),
                model_pruning=self.model_pruning
            )
        else:
            raise ValueError('Unsupported model type for simplified eval_AP (YOLOv3/Tiny only).')

        return eval_model

    def update_eval_model(self, train_model):
        """
        학습 모델 가중치를 eval_model로 복사
        (구조 동일하다고 가정, weight만 공유)
        """
        tmp_weights_path = os.path.join(
            tempfile.gettempdir(),
            str(random.randint(10, 1000000)) + '.h5'
        )
        train_model.save_weights(tmp_weights_path)
        self.eval_model.load_weights(tmp_weights_path)
        os.remove(tmp_weights_path)

        if self.model_pruning:
            eval_model = sparsity.strip_pruning(self.eval_model)
        else:
            eval_model = self.eval_model

        return eval_model

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.eval_epoch_interval != 0:
            return

        # 일정 epoch마다 eval 수행
        eval_model = self.update_eval_model(self.model)
        mAP = eval_AP(
            eval_model,
            'H5',
            self.annotation_lines,
            self.anchors,
            self.class_names,
            self.model_input_shape,
            eval_type='VOC',
            iou_threshold=0.5,
            nms_iou_threshold=0.5,
            conf_threshold=0.001,
            elim_grid_sense=self.elim_grid_sense,
            save_result=False
        )

        if self.save_eval_checkpoint and mAP > self.best_mAP:
            self.best_mAP = mAP
            self.model.save(
                os.path.join(
                    self.log_dir,
                    'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-mAP{mAP:.3f}.h5'
                    .format(
                        epoch=(epoch + 1),
                        loss=logs.get('loss'),
                        val_loss=logs.get('val_loss'),
                        mAP=mAP
                    )
                )
            )


############################################################
# 1) Annotation Parsing / GT 처리
############################################################
def annotation_parse(annotation_lines, class_names):
    """
    annotation txt → 이미지별 GT / 클래스별 GT dict 생성
    """
    annotation_records = OrderedDict()
    classes_records = OrderedDict({cn: [] for cn in class_names})

    for line in annotation_lines:
        box_records = {}
        img = line.split(' ')[0]
        boxes = line.split(' ')[1:]
        for box in boxes:
            cname = class_names[int(box.split(',')[-1])]
            coord = ','.join(box.split(',')[:-1])
            box_records[coord] = cname

            classes_records[cname].append([os.path.basename(img), coord])
        annotation_records[img] = box_records

    return annotation_records, classes_records


def transform_gt_record(gt_records, class_names):
    """
    (xmin,ymin,xmax,ymax:class) 형식을 (boxes, classes, scores)로 변환
    """
    if gt_records is None or len(gt_records) == 0:
        return [], [], []

    gt_boxes, gt_classes, gt_scores = [], [], []
    for coord, cname in gt_records.items():
        gt_boxes.append([int(x) for x in coord.split(',')])
        gt_classes.append(class_names.index(cname))
        gt_scores.append(1.0)

    return np.array(gt_boxes), np.array(gt_classes), np.array(gt_scores)


############################################################
# 2) YOLOv3 / Tiny-YOLOv3 전용 모델 예측
############################################################
def yolo_predict_keras(model,
                       image,
                       anchors,
                       num_classes,
                       model_input_shape,
                       nms_iou_threshold,
                       conf_threshold,
                       elim_grid_sense):
    """
    H5 YOLOv3 / Tiny YOLOv3 모델 전용 예측 함수
    """
    image_data = preprocess_image(image, model_input_shape)
    image_shape = image.size[::-1]

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


############################################################
# 3) 예측 결과를 Pascal VOC 포맷으로 변환
############################################################
def get_prediction_class_records(model,
                                 model_format,
                                 annotation_records,
                                 anchors,
                                 class_names,
                                 model_input_shape,
                                 nms_iou_threshold,
                                 conf_threshold,
                                 elim_grid_sense,
                                 save_result):

    if model_format != 'H5':
        raise ValueError("Only H5 format supported in this simplified eval_AP module.")

    os.makedirs('result', exist_ok=True)
    result_file = open('result/detection_result.txt', 'w')

    pred_classes_records = OrderedDict()
    pbar = tqdm(total=len(annotation_records), desc="Eval model")

    for (img_path, gt_records) in annotation_records.items():
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image, dtype='uint8')

        pred_boxes, pred_classes, pred_scores = yolo_predict_keras(
            model,
            image,
            anchors,
            len(class_names),
            model_input_shape,
            nms_iou_threshold,
            conf_threshold,
            elim_grid_sense
        )

        pbar.update(1)
        result_file.write(img_path)
        for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
            xmin, ymin, xmax, ymax = box
            result_file.write(f" {xmin},{ymin},{xmax},{ymax},{cls},{score}")
        result_file.write("\n")
        result_file.flush()

        if save_result:
            gt_boxes, gt_classes, gt_scores = transform_gt_record(gt_records, class_names)

            result_dir = "result/detection"
            os.makedirs(result_dir, exist_ok=True)
            colors = get_colors(len(class_names))

            image_np = draw_boxes(
                image_np,
                gt_boxes,
                gt_classes,
                gt_scores,
                class_names,
                colors=None,
                show_score=False
            )
            image_np = draw_boxes(
                image_np,
                pred_boxes,
                pred_classes,
                pred_scores,
                class_names,
                colors
            )

            Image.fromarray(image_np).save(
                os.path.join(result_dir, os.path.basename(img_path))
            )

        # VOC-style records
        for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
            cname = class_names[cls]
            coord = "{},{},{},{}".format(*box)

            if cname not in pred_classes_records:
                pred_classes_records[cname] = []
            pred_classes_records[cname].append(
                [os.path.basename(img_path), coord, score]
            )

    pbar.close()
    result_file.close()

    for lst in pred_classes_records.values():
        lst.sort(key=lambda x: x[2], reverse=True)

    return pred_classes_records


############################################################
# 4) IoU, AP 계산 (Pascal VOC 규격)
############################################################
def box_iou(pred_box, gt_box):
    """
    (xmin,ymin,xmax,ymax) IoU 계산
    """
    inter_x1 = max(pred_box[0], gt_box[0])
    inter_y1 = max(pred_box[1], gt_box[1])
    inter_x2 = min(pred_box[2], gt_box[2])
    inter_y2 = min(pred_box[3], gt_box[3])

    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h

    pred_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    gt_area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)

    return 0 if (pred_area + gt_area - inter_area) == 0 else \
        inter_area / (pred_area + gt_area - inter_area)


def match_gt_box(pred_record, gt_records, iou_threshold=0.5):
    """
    prediction 1개에 대해 가장 IoU 높은 GT 매칭
    """
    pred_box = [float(x) for x in pred_record[1].split(',')]
    best_iou, best_idx = 0, -1

    for i, gt in enumerate(gt_records):
        gt_box = [float(x) for x in gt[1].split(',')]
        iou = box_iou(pred_box, gt_box)
        if iou > best_iou and gt[2] == 'unused' and pred_record[0] == gt[0]:
            best_iou, best_idx = iou, i

    return best_idx if best_iou >= iou_threshold else -1


def voc_ap(rec, prec):
    """
    Pascal VOC 2012 규격 AP 계산
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = [i for i in range(1, len(mrec)) if mrec[i] != mrec[i - 1]]

    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap, mrec, mpre


def get_rec_prec(tp, fp, gt_records):
    """
    recall / precision 계산
    """
    for i in range(1, len(tp)):
        tp[i] += tp[i - 1]
        fp[i] += fp[i - 1]

    rec = [tp[i] / len(gt_records) if len(gt_records) > 0 else 0 for i in range(len(tp))]
    prec = [tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0 for i in range(len(tp))]
    return rec, prec


############################################################
# 5) class 단위 AP 계산
############################################################
def calc_AP(gt_records, pred_records, class_name, iou_threshold, show_result=False):
    """
    한 클래스의 AP 계산
    """
    gt_records = [rec + ['unused'] for rec in gt_records]

    tp = [0] * len(pred_records)
    fp = [0] * len(pred_records)

    for i, pred in enumerate(pred_records):
        img = pred[0]
        img_gt = [g for g in gt_records if g[0] == img]

        idx = match_gt_box(pred, img_gt, iou_threshold)
        if idx >= 0:
            img_gt[idx][2] = 'used'
            tp[i] = 1
        else:
            fp[i] = 1

    rec, prec = get_rec_prec(tp, fp, gt_records)
    ap, _, _ = voc_ap(rec, prec)

    return ap


############################################################
# 6) Pascal VOC mAP 계산
############################################################
def get_mean_metric(metric_records, gt_classes_records):
    """
    GT가 존재하는 클래스에 대해서만 mean metric 계산
    """
    vals = []
    for cname, metric in metric_records.items():
        if cname in gt_classes_records and len(gt_classes_records[cname]) > 0:
            vals.append(metric)
    return np.mean(vals) * 100 if len(vals) > 0 else 0.0


def compute_mAP_PascalVOC(annotation_records,
                           gt_classes_records,
                           pred_classes_records,
                           class_names,
                           iou_threshold):
    """
    Pascal VOC mAP 계산 (YOLOv3 전용 간소화 버전)
    """
    APs = {}
    for cname in class_names:
        if len(gt_classes_records[cname]) == 0:
            APs[cname] = 0
            continue

        if cname not in pred_classes_records:
            APs[cname] = 0
            continue

        APs[cname] = calc_AP(
            gt_classes_records[cname],
            pred_classes_records[cname],
            cname,
            iou_threshold
        )

    mAP = get_mean_metric(APs, gt_classes_records)
    return mAP, APs


############################################################
# 7) 최종 eval_AP 엔트리
############################################################
def eval_AP(model,
            model_format,
            annotation_lines,
            anchors,
            class_names,
            model_input_shape,
            eval_type,
            iou_threshold,
            nms_iou_threshold,
            conf_threshold,
            elim_grid_sense,
            save_result,
            class_filter=None):
    """
    YOLOv3/Tiny YOLOv3 전용 eval_AP
    (VOC만 지원, v2/v5 완전 제거)
    """
    annotation_records, gt_classes_records = annotation_parse(annotation_lines, class_names)

    pred_classes_records = get_prediction_class_records(
        model,
        model_format,
        annotation_records,
        anchors,
        class_names,
        model_input_shape,
        nms_iou_threshold,
        conf_threshold,
        elim_grid_sense,
        save_result
    )

    if eval_type == 'VOC':
        mAP, APs = compute_mAP_PascalVOC(
            annotation_records,
            gt_classes_records,
            pred_classes_records,
            class_names,
            iou_threshold
        )
        return mAP

    elif eval_type == '':
        return None

    else:
        raise ValueError("Only VOC eval supported in simplified YOLOv3-only eval_AP.")
