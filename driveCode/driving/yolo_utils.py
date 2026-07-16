import cv2
import numpy as np


def letterbox_image(image, size):
    image_h, image_w, _ = image.shape
    target_w, target_h = size
    scale = min(target_w / image_w, target_h / image_h)
    resized_w, resized_h = int(image_w * scale), int(image_h * scale)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    boxed = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    y_start, x_start = (target_h - resized_h) // 2, (target_w - resized_w) // 2
    boxed[y_start:y_start + resized_h, x_start:x_start + resized_w] = resized
    return boxed


def pre_process(image, model_image_size):
    rgb_image = image[..., ::-1]
    boxed = letterbox_image(rgb_image, tuple(reversed(model_image_size)))
    return np.expand_dims(np.asarray(boxed, dtype=np.float32) / 255.0, 0)


def _get_feats(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = np.reshape(np.asarray(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
    grid_size = np.shape(feats)[1:3]
    predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
    grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
    grid = np.asarray(np.concatenate([grid_x, grid_y], axis=-1), dtype=np.float32)
    box_xy = (1 / (1 + np.exp(-predictions[..., :2])) + grid) / np.asarray(grid_size[::-1], dtype=np.float32)
    box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.asarray(input_shape[::-1], dtype=np.float32)
    box_confidence = 1 / (1 + np.exp(-predictions[..., 4:5]))
    box_class_probs = 1 / (1 + np.exp(-predictions[..., 5:]))
    return box_xy, box_wh, box_confidence, box_class_probs


def _correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.asarray(input_shape, dtype=np.float32)
    image_shape = np.asarray(image_shape, dtype=np.float32)
    new_shape = np.around(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2.0 / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    boxes = np.concatenate([
        box_yx[..., 0:1] - box_hw[..., 0:1] / 2,
        box_yx[..., 1:2] - box_hw[..., 1:2] / 2,
        box_yx[..., 0:1] + box_hw[..., 0:1] / 2,
        box_yx[..., 1:2] + box_hw[..., 1:2] / 2,
    ], axis=-1)
    return boxes * np.concatenate([image_shape, image_shape], axis=-1)


def _boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = _get_feats(feats, anchors, num_classes, input_shape)
    boxes = np.reshape(_correct_boxes(box_xy, box_wh, input_shape, image_shape), [-1, 4])
    scores = np.reshape(box_confidence * box_class_probs, [-1, num_classes])
    return boxes, scores


def nms_boxes(boxes, scores, iou_threshold=0.1):
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        index = order[0]
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width * height
        union = areas[index] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def evaluate(yolo_outputs, image_shape, class_names, anchors, threshold, max_boxes=20):
    anchor_mask = [[3, 4, 5], [0, 1, 2]]
    input_shape = np.shape(yolo_outputs[0])[1:3] * np.asarray([32, 32])
    boxes, scores = [], []
    for index, output in enumerate(yolo_outputs):
        current_boxes, current_scores = _boxes_and_scores(
            output, anchors[anchor_mask[index]], len(class_names), input_shape, image_shape
        )
        boxes.append(current_boxes)
        scores.append(current_scores)
    boxes = np.concatenate(boxes, axis=0)
    scores = np.concatenate(scores, axis=0)

    final_boxes, final_scores, final_classes = [], [], []
    for class_index in range(len(class_names)):
        mask = scores[:, class_index] >= threshold
        class_boxes = boxes[mask]
        class_scores = scores[mask, class_index]
        keep = nms_boxes(class_boxes, class_scores)[:max_boxes]
        final_boxes.append(class_boxes[keep])
        final_scores.append(class_scores[keep])
        final_classes.append(np.full(len(keep), class_index, dtype=np.int32))
    return (
        np.concatenate(final_boxes, axis=0),
        np.concatenate(final_scores, axis=0),
        np.concatenate(final_classes, axis=0),
    )
