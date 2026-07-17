# python train_yolov3.py   --anchors_path configs/tiny_yolo3_anchors.txt  --classes_path configs/lane_class.txt  --annotation_file data/lane_detection/train/_annotations.txt  --model_input_shape 256x256   --batch_size 16
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiny YOLOv3 Training Script (Darknet backbone, lane detection)

- 모델 타입: tiny_yolo3_darknet (고정)
- Dataset: lane_detection (annotation txt)
- Anchors: tiny_yolo3_anchors.txt (6 anchors)
- Classes: lane_class.txt

"""

import os
import time
import argparse
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (
    TensorBoard, ModelCheckpoint, ReduceLROnPlateau,
    EarlyStopping, TerminateOnNaN, Callback
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from tensorflow_model_optimization.sparsity import keras as sparsity

from yolo3.model import get_yolo3_train_model
from yolo3.data import yolo3_data_generator_wrapper
from common.utils import get_classes, get_anchors, get_dataset, optimize_tf_gpu
from common.callbacks import CheckpointCleanCallBack


# ===== 환경 세팅 =====
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

optimize_tf_gpu(tf, K)


# =====================================================
# 1. 커스텀 콜백: EpochTracker & BestMetricCheckpoint
# =====================================================
class EpochTracker(Callback):
    """전체 학습 동안 누적 epoch 번호를 추적"""
    def __init__(self):
        super().__init__()
        self.epoch = 0  # 1, 2, 3, ...

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1


class BestMetricCheckpoint(Callback):
    """
    monitor(metric) 기준으로 가장 좋은 모델을 항상 따로 저장하는 콜백.
    - 기본 monitor: 'val_loss'
    - mode: 'min' (loss), 'max' (mAP) 등 선택 가능
    """
    def __init__(self, log_dir, epoch_tracker,
                 monitor='val_loss', mode='min',
                 filename_prefix='best'):
        super().__init__()
        self.log_dir = log_dir
        self.epoch_tracker = epoch_tracker
        self.monitor = monitor
        self.mode = mode
        self.filename_prefix = filename_prefix

        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")

        if mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf

        self.best_path = None

    def is_better(self, current):
        if self.mode == 'min':
            return current < self.best
        else:
            return current > self.best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor, None)
        if current is None:
            # 모니터링할 metric이 없는 경우 (예: 아직 val_loss 계산 안됨)
            return

        global_epoch = self.epoch_tracker.epoch + 1  # on_epoch_end 직전 기준

        if self.is_better(current):
            self.best = current
            filename = f"{self.filename_prefix}_{self.monitor}_epoch{global_epoch:03d}_{current:.4f}.h5"
            save_path = os.path.join(self.log_dir, filename)
            # 현재 train model(self.model)을 저장
            self.model.save(save_path)
            self.best_path = save_path
            print(
                f"[BestMetricCheckpoint] New best {self.monitor}={current:.6f} "
                f"at epoch {global_epoch}, saved to {save_path}"
            )


# =====================================================
# 2. Tiny YOLOv3 Training Main
# =====================================================
def main(args):
    # ----- 기본 설정 -----
    log_dir = os.path.join('logs', 'tiny_yolov3')
    os.makedirs(log_dir, exist_ok=True)

    class_names = get_classes(args.classes_path)
    num_classes = len(class_names)

    anchors = get_anchors(args.anchors_path)
    num_anchors = len(anchors)

    # tiny-yolov3는 6 anchors, 2 feature layers
    assert num_anchors == 6, "Tiny YOLOv3는 6개의 anchor를 사용해야 합니다."

    # ----- 데이터 로드 -----
    dataset = get_dataset(args.annotation_file)

    if args.val_annotation_file:
        val_dataset = get_dataset(args.val_annotation_file)
        num_train = len(dataset)
        num_val = len(val_dataset)
        dataset.extend(val_dataset)
    else:
        val_split = args.val_split
        num_val = int(len(dataset) * val_split)
        num_train = len(dataset) - num_val

    train_dataset = dataset[:num_train]
    val_dataset = dataset[num_train:]

    # ----- 입력 크기 -----
    height, width = args.model_input_shape.split('x')
    input_shape = (int(height), int(width))
    assert input_shape[0] % 32 == 0 and input_shape[1] % 32 == 0, \
        "model_input_shape는 32의 배수여야 합니다."

    # ----- multiscale / augment 설정 (간단 버전: 사용 안함) -----
    rescale_interval = -1       # multiscale X
    enhance_augment = None      # mosaic X
    multi_anchor_assign = False

    # ----- 콜백들 준비 -----
    logging = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        update_freq='batch'
    )

    # 기본 Keras 체크포인트 (val_loss 최소 기준)
    checkpoint = ModelCheckpoint(
        os.path.join(
            log_dir,
            'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
        ),
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        save_freq='epoch'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        mode='min',
        patience=10,
        verbose=1,
        cooldown=0,
        min_lr=1e-10
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=50,
        verbose=1,
        mode='min'
    )

    checkpoint_clean = CheckpointCleanCallBack(
        log_dir, max_val_keep=5, max_eval_keep=2
    )

    terminate_on_nan = TerminateOnNaN()

    # epoch 추적 콜백
    epoch_tracker = EpochTracker()

    # Best metric checkpoint 콜백
    best_metric_ckpt = BestMetricCheckpoint(
        log_dir=log_dir,
        epoch_tracker=epoch_tracker,
        monitor='val_loss',  # 나중에 mAP metric 추가 시 'val_mAP'로 변경 가능
        mode='min',          # mAP일 경우 'max'
        filename_prefix='best'
    )

    callbacks = [
        logging,
        checkpoint,
        reduce_lr,
        early_stopping,
        terminate_on_nan,
        checkpoint_clean,
        epoch_tracker,
        best_metric_ckpt,
    ]

    # ----- Optimizer & Pruning 설정 -----
    steps_per_epoch = max(1, num_train // args.batch_size)
    pruning_end_step = steps_per_epoch * args.total_epoch if args.model_pruning else 0

    optimizer = Adam(
        lr=args.learning_rate,
        decay=0.0
    )

    # ----- Tiny YOLOv3 Train Model 생성 -----
    # model_type을 tiny_yolo3_darknet으로 고정
    model_type = 'tiny_yolo3_darknet'

    model, model_body = get_yolo3_train_model(
    model_type=model_type,
    anchors=anchors,
    num_classes=num_classes,
    weights_path=args.weights_path,
    freeze_level=args.freeze_level,
    optimizer=optimizer,
    label_smoothing=args.label_smoothing,
    elim_grid_sense=args.elim_grid_sense,
    model_pruning=args.model_pruning,
    pruning_end_step=pruning_end_step
)

    model.summary()

    # ----- Data Generator 구성 -----
    train_gen = yolo3_data_generator_wrapper(
        train_dataset, args.batch_size, input_shape,
        anchors, num_classes, enhance_augment,
        rescale_interval, multi_anchor_assign=multi_anchor_assign
    )

    val_gen = yolo3_data_generator_wrapper(
        val_dataset, args.batch_size, input_shape,
        anchors, num_classes, None,
        -1, multi_anchor_assign=multi_anchor_assign
    )

    # =====================================================
    # 3. Training Loop + KeyboardInterrupt 처리
    # =====================================================
    initial_epoch = args.init_epoch
    transfer_epochs = args.transfer_epoch
    total_epochs = args.total_epoch

    try:
        # ----- 1단계: freeze 상태 transfer training -----
        if transfer_epochs > 0:
            print("=== Transfer training (frozen) stage ===")
            print(
                'Train on {} samples, val on {} samples, batch size {}, '
                'input_shape {}.'.format(
                    num_train, num_val, args.batch_size, input_shape
                )
            )

            model.fit_generator(
                train_gen,
                steps_per_epoch=max(1, num_train // args.batch_size),
                validation_data=val_gen,
                validation_steps=max(1, num_val // args.batch_size),
                epochs=initial_epoch + transfer_epochs,
                initial_epoch=initial_epoch,
                workers=1,
                use_multiprocessing=False,
                max_queue_size=10,
                callbacks=callbacks
            )

            # 2단계 진입 전 살짝 대기
            time.sleep(2)

            # ----- Freeze 풀고 전체 fine-tune -----
            print("Unfreeze and continue training, to fine-tune.")
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(
                optimizer=optimizer,
                loss={'yolo_loss': lambda y_true, y_pred: y_pred}
            )

            print(
                'Train on {} samples, val on {} samples, batch size {}, '
                'input_shape {}.'.format(
                    num_train, num_val, args.batch_size, input_shape
                )
            )

            model.fit_generator(
                train_gen,
                steps_per_epoch=max(1, num_train // args.batch_size),
                validation_data=val_gen,
                validation_steps=max(1, num_val // args.batch_size),
                epochs=total_epochs,
                initial_epoch=initial_epoch + transfer_epochs,
                workers=1,
                use_multiprocessing=False,
                max_queue_size=10,
                callbacks=callbacks
            )

        else:
            # freeze 단계 없이 바로 full training
            print("=== Full training (no frozen stage) ===")
            print(
                'Train on {} samples, val on {} samples, batch size {}, '
                'input_shape {}.'.format(
                    num_train, num_val, args.batch_size, input_shape
                )
            )

            model.fit_generator(
                train_gen,
                steps_per_epoch=max(1, num_train // args.batch_size),
                validation_data=val_gen,
                validation_steps=max(1, num_val // args.batch_size),
                epochs=total_epochs,
                initial_epoch=initial_epoch,
                workers=1,
                use_multiprocessing=False,
                max_queue_size=10,
                callbacks=callbacks
            )

    except KeyboardInterrupt:
        print("\n[WARN] Training interrupted by user (Ctrl+C).")
        current_epoch = epoch_tracker.epoch
        print(f"[INFO] Last finished epoch: {current_epoch}")

        # --- 1) 이미 가지고 있는 model_body로 inference 모델 생성 ---
        if model_body is not None:
            inf_model = Model(
                inputs=model_body.input,
                outputs=model_body.output,
                name="tiny_yolov3_inference"
            )

            inf_path = os.path.join(
                log_dir,
                f"inference_tiny_yolov3_epoch{current_epoch:03d}_interrupt.h5"
            )
            inf_model.save(inf_path)
            print(f"[INFO] Interrupt inference model saved to: {inf_path}")
        else:
            print("[ERROR] model_body is None. Interrupt-save skipped.")

        # --- 2) 현재까지의 BEST checkpoint 정보 출력 ---
        if best_metric_ckpt.best_path is not None:
            print(
                f"[INFO] Best checkpoint so far "
                f"({best_metric_ckpt.monitor}={best_metric_ckpt.best:.6f}):"
            )
            print(f"       {best_metric_ckpt.best_path}")
        else:
            print("[INFO] No best checkpoint recorded yet (monitor metric might be missing).")

        return

    # =====================================================
    # 4. 정상 종료 시: pruning 제거 + 최종 모델 저장
    # =====================================================
    if args.model_pruning:
        model = sparsity.strip_pruning(model)

    final_train_path = os.path.join(log_dir, 'trained_final.h5')
    model.save(final_train_path)
    print(f"[INFO] Final train model saved to: {final_train_path}")

    # --- 최종 inference 모델도 저장 ---
    model_body = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            model_body = layer
            break
    if model_body is None and len(model.layers) > 1 and isinstance(model.layers[1], tf.keras.Model):
        model_body = model.layers[1]

    if model_body is not None:
        final_epoch = epoch_tracker.epoch
        inf_model = Model(
            inputs=model_body.input,
            outputs=model_body.output,
            name="tiny_yolov3_inference"
        )
        final_inf_path = os.path.join(
            log_dir,
            f"inference_tiny_yolov3_final_epoch{final_epoch:03d}.h5"
        )
        inf_model.save(final_inf_path)
        print(f"[INFO] Final inference model saved to: {final_inf_path}")
    else:
        print("[WARN] model_body is None at normal end; inference model not saved.")

    # BEST checkpoint 정보 출력
    if best_metric_ckpt.best_path is not None:
        print(
            f"[INFO] Best checkpoint over whole training "
            f"({best_metric_ckpt.monitor}={best_metric_ckpt.best:.6f}):"
        )
        print(f"       {best_metric_ckpt.best_path}")
    else:
        print("[INFO] No best checkpoint was saved (monitor metric may be missing).")


# =====================================================
# 5. Argument Parser
# =====================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 필수 경로 인자들
    parser.add_argument('--anchors_path', type=str, required=True,
                        help='path to tiny yolo3 anchor definitions')
    parser.add_argument('--classes_path', type=str, required=True,
                        help='path to class definitions')
    parser.add_argument('--annotation_file', type=str, required=True,
                        help='train annotation txt file (yolo format)')
    parser.add_argument('--val_annotation_file', type=str, default=None,
                        help='optional val annotation txt file')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='if no val_annotation_file, split ratio for validation')

    parser.add_argument('--model_input_shape', type=str, default='256x256',
                        help="model input shape as <height>x<width>, multiples of 32")
    parser.add_argument('--weights_path', type=str, default=None,
                        help="(주의) tiny_yolo3_darknet에서는 현재 무시됨")

    # Training 설정
    parser.add_argument('--batch_size', type=int, default=16,
                        help="batch size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="initial learning rate")
    parser.add_argument('--transfer_epoch', type=int, default=20,
                        help="frozen(backbone) training epochs")
    parser.add_argument('--freeze_level', type=int, default=1, choices=[0, 1, 2],
                        help="0: no freeze, 1: freeze backbone, 2: freeze all but last layers")
    parser.add_argument('--init_epoch', type=int, default=0,
                        help="initial epoch index")
    parser.add_argument('--total_epoch', type=int, default=250,
                        help="total training epochs")
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help="label smoothing factor")
    parser.add_argument('--elim_grid_sense', action='store_true',
                        help="eliminate grid sensitivity")
    parser.add_argument('--model_pruning', action='store_true',
                        help="enable model pruning (TF1.x style)")

    args = parser.parse_args()
    main(args)
