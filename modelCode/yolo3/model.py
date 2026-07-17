#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create YOLOv3/v4 models with different backbone & head

여기서는 tiny_yolo3_darknet 학습에 필요한 부분만 최소한으로 정리:
 - yolo3_tiny_model_map 에서 tiny_yolo3_darknet → tiny_yolo3_body 사용
 - tiny_yolo3_* 에 대해서는 get_yolo3_train_model에서 weights_path를 무시(에러 방지)
"""

import os
from functools import partial

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from yolo3.models.yolo3_darknet import (
    yolo3_body,
    yolo3lite_body,
    tiny_yolo3_body,
    tiny_yolo3lite_body,
    custom_yolo3_spp_body,
)
from yolo3.models.yolo3_mobilenet import (
    yolo3_mobilenet_body,
    tiny_yolo3_mobilenet_body,
    yolo3lite_mobilenet_body,
    yolo3lite_spp_mobilenet_body,
    tiny_yolo3lite_mobilenet_body,
)
from yolo3.models.yolo3_mobilenetv2 import (
    yolo3_mobilenetv2_body,
    tiny_yolo3_mobilenetv2_body,
    yolo3lite_mobilenetv2_body,
    yolo3lite_spp_mobilenetv2_body,
    tiny_yolo3lite_mobilenetv2_body,
    yolo3_ultralite_mobilenetv2_body,
    tiny_yolo3_ultralite_mobilenetv2_body,
)
from yolo3.models.yolo3_shufflenetv2 import (
    yolo3_shufflenetv2_body,
    tiny_yolo3_shufflenetv2_body,
    yolo3lite_shufflenetv2_body,
    yolo3lite_spp_shufflenetv2_body,
    tiny_yolo3lite_shufflenetv2_body,
)
from yolo3.models.yolo3_vgg16 import yolo3_vgg16_body, tiny_yolo3_vgg16_body
from yolo3.models.yolo3_xception import (
    yolo3_xception_body,
    yolo3lite_xception_body,
    tiny_yolo3_xception_body,
    tiny_yolo3lite_xception_body,
    yolo3_spp_xception_body,
)
from yolo3.models.yolo3_nano import yolo3_nano_body
from yolo3.models.yolo3_efficientnet import (
    yolo3_efficientnet_body,
    tiny_yolo3_efficientnet_body,
    yolo3lite_efficientnet_body,
    yolo3lite_spp_efficientnet_body,
    tiny_yolo3lite_efficientnet_body,
)
from yolo3.models.yolo3_mobilenetv3_large import (
    yolo3_mobilenetv3large_body,
    yolo3lite_mobilenetv3large_body,
    tiny_yolo3_mobilenetv3large_body,
    tiny_yolo3lite_mobilenetv3large_body,
)
from yolo3.models.yolo3_mobilenetv3_small import (
    yolo3_mobilenetv3small_body,
    yolo3lite_mobilenetv3small_body,
    tiny_yolo3_mobilenetv3small_body,
    tiny_yolo3lite_mobilenetv3small_body,
    yolo3_ultralite_mobilenetv3small_body,
    tiny_yolo3_ultralite_mobilenetv3small_body,
)
from yolo3.models.yolo3_peleenet import (
    yolo3_peleenet_body,
    yolo3lite_peleenet_body,
    tiny_yolo3_peleenet_body,
    tiny_yolo3lite_peleenet_body,
    yolo3_ultralite_peleenet_body,
    tiny_yolo3_ultralite_peleenet_body,
)
from yolo3.models.yolo3_ghostnet import (
    yolo3_ghostnet_body,
    yolo3lite_ghostnet_body,
    tiny_yolo3_ghostnet_body,
    tiny_yolo3lite_ghostnet_body,
    yolo3_ultralite_ghostnet_body,
    tiny_yolo3_ultralite_ghostnet_body,
)


from yolo3.loss import yolo3_loss
from yolo3.postprocess import batched_yolo3_postprocess
from common.model_utils import add_metrics, get_pruning_model

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


# =====================================================
# 1. Model map 정의
# =====================================================

# YOLOv3 full-size
yolo3_model_map = {
    'yolo3_mobilenet': [yolo3_mobilenet_body, 87, None],
    'yolo3_mobilenet_lite': [yolo3lite_mobilenet_body, 87, None],
    'yolo3_mobilenet_lite_spp': [yolo3lite_spp_mobilenet_body, 87, None],
    'yolo3_mobilenetv2': [yolo3_mobilenetv2_body, 155, None],
    'yolo3_mobilenetv2_lite': [yolo3lite_mobilenetv2_body, 155, None],
    'yolo3_mobilenetv2_lite_spp': [yolo3lite_spp_mobilenetv2_body, 155, None],
    'yolo3_mobilenetv2_ultralite': [yolo3_ultralite_mobilenetv2_body, 155, None],

    'yolo3_mobilenetv3large': [yolo3_mobilenetv3large_body, 195, None],
    'yolo3_mobilenetv3large_lite': [yolo3lite_mobilenetv3large_body, 195, None],
    'yolo3_mobilenetv3small': [yolo3_mobilenetv3small_body, 166, None],
    'yolo3_mobilenetv3small_lite': [yolo3lite_mobilenetv3small_body, 166, None],
    'yolo3_mobilenetv3small_ultralite': [yolo3_ultralite_mobilenetv3small_body, 166, None],

    'yolo3_peleenet': [yolo3_peleenet_body, 366, None],
    'yolo3_peleenet_lite': [yolo3lite_peleenet_body, 366, None],
    'yolo3_peleenet_ultralite': [yolo3_ultralite_peleenet_body, 366, None],

    'yolo3_ghostnet': [yolo3_ghostnet_body, 292, None],
    'yolo3_ghostnet_lite': [yolo3lite_ghostnet_body, 292, None],
    'yolo3_ghostnet_ultralite': [yolo3_ultralite_ghostnet_body, 292, None],

    'yolo3_shufflenetv2': [yolo3_shufflenetv2_body, 205, None],
    'yolo3_shufflenetv2_lite': [yolo3lite_shufflenetv2_body, 205, None],
    'yolo3_shufflenetv2_lite_spp': [yolo3lite_spp_shufflenetv2_body, 205, None],

    # NOTE: backbone_length is for EfficientNetB3
    'yolo3_efficientnet': [yolo3_efficientnet_body, 382, None],
    'yolo3_efficientnet_lite': [yolo3lite_efficientnet_body, 382, None],
    'yolo3_efficientnet_lite_spp': [yolo3lite_spp_efficientnet_body, 382, None],

    'yolo3_darknet': [yolo3_body, 185, os.path.join(ROOT_PATH, 'weights', 'darknet53.h5')],
    'yolo3_darknet_spp': [custom_yolo3_spp_body, 185, os.path.join(ROOT_PATH, 'weights', 'yolov3-spp.h5')],
    'yolo3_darknet_lite': [yolo3lite_body, 0, None],
    'yolo3_vgg16': [yolo3_vgg16_body, 19, None],
    'yolo3_xception': [yolo3_xception_body, 132, None],
    'yolo3_xception_lite': [yolo3lite_xception_body, 132, None],
    'yolo3_xception_spp': [yolo3_spp_xception_body, 132, None],

    'yolo3_nano': [yolo3_nano_body, 268, None],
}


# Tiny YOLOv3 / Tiny YOLOv4
yolo3_tiny_model_map = {
    'tiny_yolo3_mobilenet': [tiny_yolo3_mobilenet_body, 87, None],
    'tiny_yolo3_mobilenet_lite': [tiny_yolo3lite_mobilenet_body, 87, None],
    'tiny_yolo3_mobilenetv2': [tiny_yolo3_mobilenetv2_body, 155, None],
    'tiny_yolo3_mobilenetv2_lite': [tiny_yolo3lite_mobilenetv2_body, 155, None],
    'tiny_yolo3_mobilenetv2_ultralite': [tiny_yolo3_ultralite_mobilenetv2_body, 155, None],

    'tiny_yolo3_mobilenetv3large': [tiny_yolo3_mobilenetv3large_body, 195, None],
    'tiny_yolo3_mobilenetv3large_lite': [tiny_yolo3lite_mobilenetv3large_body, 195, None],
    'tiny_yolo3_mobilenetv3small': [tiny_yolo3_mobilenetv3small_body, 166, None],
    'tiny_yolo3_mobilenetv3small_lite': [tiny_yolo3lite_mobilenetv3small_body, 166, None],
    'tiny_yolo3_mobilenetv3small_ultralite': [tiny_yolo3_ultralite_mobilenetv3small_body, 166, None],

    'tiny_yolo3_peleenet': [tiny_yolo3_peleenet_body, 366, None],
    'tiny_yolo3_peleenet_lite': [tiny_yolo3lite_peleenet_body, 366, None],
    'tiny_yolo3_peleenet_ultralite': [tiny_yolo3_ultralite_peleenet_body, 366, None],

    'tiny_yolo3_ghostnet': [tiny_yolo3_ghostnet_body, 292, None],
    'tiny_yolo3_ghostnet_lite': [tiny_yolo3lite_ghostnet_body, 292, None],
    'tiny_yolo3_ghostnet_ultralite': [tiny_yolo3_ultralite_ghostnet_body, 292, None],

    'tiny_yolo3_shufflenetv2': [tiny_yolo3_shufflenetv2_body, 205, None],
    'tiny_yolo3_shufflenetv2_lite': [tiny_yolo3lite_shufflenetv2_body, 205, None],

    # NOTE: backbone_length is for EfficientNetB0
    'tiny_yolo3_efficientnet': [tiny_yolo3_efficientnet_body, 235, None],
    'tiny_yolo3_efficientnet_lite': [tiny_yolo3lite_efficientnet_body, 235, None],

    # 🔑 여기: tiny_yolo3_darknet은 tiny_yolo3_body 그대로 사용 (pretrained weight 없음)
    'tiny_yolo3_darknet': [tiny_yolo3_body, 20, None],

    'tiny_yolo3_darknet_lite': [tiny_yolo3lite_body, 0, None],
    'tiny_yolo3_vgg16': [tiny_yolo3_vgg16_body, 19, None],
    'tiny_yolo3_xception': [tiny_yolo3_xception_body, 132, None],
    'tiny_yolo3_xception_lite': [tiny_yolo3lite_xception_body, 132, None],
}


# =====================================================
# 2. 모델 생성 함수
# =====================================================
def get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes,
                    input_tensor=None, input_shape=None,
                    model_pruning=False, pruning_end_step=10000):
    # 입력 텐서 준비
    if input_shape:
        input_tensor = Input(shape=input_shape, name='image_input')

    if input_tensor is None:
        input_tensor = Input(shape=(None, None, 3), name='image_input')

    # Tiny YOLOv3 (6 anchors, 2 feature layers)
    if num_feature_layers == 2:
        if model_type in yolo3_tiny_model_map:
            model_function, backbone_len, weights_path = yolo3_tiny_model_map[model_type]
            # tiny는 여기서 pretrain weight를 사용하지 않도록 통일
            model_body = model_function(input_tensor, num_anchors // 2, num_classes)
        else:
            raise ValueError('This tiny model type is not supported now')

    # YOLOv3 (9 anchors, 3 feature layers)
    elif num_feature_layers == 3:
        if model_type in yolo3_model_map:
            model_function, backbone_len, weights_path = yolo3_model_map[model_type]
            if weights_path and os.path.exists(weights_path):
                print(f'Loading pretrained backbone weights from {weights_path}')
                model_body = model_function(
                    input_tensor, num_anchors // 3, num_classes,
                    weights_path=weights_path
                )
            else:
                model_body = model_function(
                    input_tensor, num_anchors // 3, num_classes
                )
        else:
            raise ValueError('This model type is not supported now')
    else:
        raise ValueError('model type mismatch anchors')

    # pruning
    if model_pruning:
        model_body = get_pruning_model(
            model_body, begin_step=0, end_step=pruning_end_step
        )

    return model_body, backbone_len


def get_yolo3_train_model(model_type, anchors, num_classes,
                          weights_path=None, freeze_level=1,
                          optimizer=Adam(lr=1e-3, decay=0),
                          label_smoothing=0,
                          elim_grid_sense=False,
                          model_pruning=False,
                          pruning_end_step=10000):
    """create the training model, for YOLOv3"""
    num_anchors = len(anchors)
    # YOLOv3: 9 anchors, 3 layer / Tiny: 6 anchors, 2 layer
    num_feature_layers = num_anchors // 3

    # y_true shape:
    # [
    #  (H/32, W/32, 3, num_classes+5),
    #  (H/16, W/16, 3, num_classes+5),
    #  (H/8,  W/8,  3, num_classes+5)
    # ]
    y_true = [
        Input(shape=(None, None, 3, num_classes + 5),
              name=f'y_true_{l}')
        for l in range(num_feature_layers)
    ]

    model_body, backbone_len = get_yolo3_model(
        model_type, num_feature_layers, num_anchors, num_classes,
        model_pruning=model_pruning, pruning_end_step=pruning_end_step
    )
    print(
        'Create {} {} model with {} anchors and {} classes.'.format(
            'Tiny' if num_feature_layers == 2 else '',
            model_type, num_anchors, num_classes
        )
    )
    print('model layer number:', len(model_body.layers))

    # -------------------------------------------------------
    # weights_path 처리:
    #  - tiny_yolo3_* 의 경우 현재 구조에서는 외부 h5(pretrained)와
    #    shape mismatch가 발생하므로 학습 시에는 로드하지 않도록 통일
    # -------------------------------------------------------
    if weights_path:
        if model_type.startswith('tiny_yolo3_'):
            print(
                f'[INFO] weights_path="{weights_path}" 가 주어졌지만 '
                f'{model_type} 에서는 학습 시 pretrained weights를 로드하지 않습니다.'
            )
        else:
            model_body.load_weights(weights_path, by_name=True)
            print(f'Load weights {weights_path}.')

    # Freeze / Unfreeze
    if freeze_level in [1, 2]:
        num = (backbone_len, len(model_body.layers) - 3)[freeze_level - 1]
        for i in range(num):
            model_body.layers[i].trainable = False
        print(
            'Freeze the first {} layers of total {} layers.'.format(
                num, len(model_body.layers)
            )
        )
    elif freeze_level == 0:
        for i in range(len(model_body.layers)):
            model_body.layers[i].trainable = True
        print('Unfreeze all of the layers.')

    # YOLO loss
    model_loss, location_loss, confidence_loss, class_loss = Lambda(
        yolo3_loss,
        name='yolo_loss',
        arguments={
            'anchors': anchors,
            'num_classes': num_classes,
            'ignore_thresh': 0.5,
            'label_smoothing': label_smoothing,
            'elim_grid_sense': elim_grid_sense,
        }
    )([*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)

    loss_dict = {
        'location_loss': location_loss,
        'confidence_loss': confidence_loss,
        'class_loss': class_loss,
    }
    add_metrics(model, loss_dict)

    model.compile(
        optimizer=optimizer,
        loss={'yolo_loss': lambda y_true, y_pred: y_pred},
    )

    return model, model_body


def get_yolo3_inference_model(model_type, anchors, num_classes,
                              weights_path=None, input_shape=None,
                              confidence=0.1, iou_threshold=0.4,
                              elim_grid_sense=False):
    """create the inference model, for YOLOv3"""
    num_anchors = len(anchors)
    num_feature_layers = num_anchors // 3

    image_shape = Input(shape=(2,), dtype='int64', name='image_shape')

    model_body, _ = get_yolo3_model(
        model_type, num_feature_layers, num_anchors,
        num_classes, input_shape=input_shape
    )
    print(
        'Create {} YOLOv3 {} model with {} anchors and {} classes.'.format(
            'Tiny' if num_feature_layers == 2 else '',
            model_type, num_anchors, num_classes
        )
    )

    if weights_path:
        model_body.load_weights(weights_path, by_name=False)
        print(f'Load weights {weights_path}.')

    boxes, scores, classes = Lambda(
        batched_yolo3_postprocess,
        name='yolo3_postprocess',
        arguments={
            'anchors': anchors,
            'num_classes': num_classes,
            'confidence': confidence,
            'iou_threshold': iou_threshold,
            'elim_grid_sense': elim_grid_sense,
        },
    )([*model_body.output, image_shape])

    model = Model([model_body.input, image_shape], [boxes, scores, classes])

    return model
