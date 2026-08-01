import cv2
import numpy as np


def pre_process(image, model_size):
    image = image[..., ::-1]
    ih, iw = image.shape[:2]
    w, h = reversed(model_size)
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    boxed = np.full((h, w, 3), 128, dtype=np.uint8)
    boxed[(h-nh)//2:(h-nh)//2+nh, (w-nw)//2:(w-nw)//2+nw] = resized
    return np.expand_dims(boxed.astype(np.float32) / 255.0, 0)


def _boxes_and_scores(feats, anchors, classes, input_shape, image_shape):
    na = len(anchors)
    grid_size = np.shape(feats)[1:3]
    pred = np.reshape(feats, [-1, grid_size[0], grid_size[1], na, classes + 5])
    gy = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
    gx = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
    grid = np.asarray(np.concatenate([gx, gy], axis=-1), dtype=np.float32)
    xy = (1.0 / (1.0 + np.exp(-pred[..., :2])) + grid) / np.asarray(grid_size[::-1], dtype=np.float32)
    wh = np.exp(pred[..., 2:4]) * np.reshape(np.asarray(anchors, dtype=np.float32), [1, 1, 1, na, 2])
    wh /= np.asarray(input_shape[::-1], dtype=np.float32)
    confidence = 1.0 / (1.0 + np.exp(-pred[..., 4:5]))
    probabilities = 1.0 / (1.0 + np.exp(-pred[..., 5:]))
    yx, hw = xy[..., ::-1], wh[..., ::-1]
    input_shape = np.asarray(input_shape, dtype=np.float32)
    image_shape = np.asarray(image_shape, dtype=np.float32)
    new_shape = np.around(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2.0 / input_shape
    scale = input_shape / new_shape
    yx = (yx - offset) * scale
    hw *= scale
    boxes = np.concatenate([yx[..., :1]-hw[..., :1]/2, yx[..., 1:]-hw[..., 1:]/2,
                            yx[..., :1]+hw[..., :1]/2, yx[..., 1:]+hw[..., 1:]/2], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return np.reshape(boxes, [-1, 4]), np.reshape(confidence * probabilities, [-1, classes])


def _nms(boxes, scores, threshold=0.1):
    if boxes.size == 0:
        return []
    y1, x1, y2, x2 = boxes.T
    areas = (y2-y1+1) * (x2-x1+1)
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i = order[0]
        keep.append(i)
        yy1, xx1 = np.maximum(y1[i], y1[order[1:]]), np.maximum(x1[i], x1[order[1:]])
        yy2, xx2 = np.minimum(y2[i], y2[order[1:]]), np.minimum(x2[i], x2[order[1:]])
        intersection = np.maximum(0, yy2-yy1+1) * np.maximum(0, xx2-xx1+1)
        iou = intersection / np.maximum(areas[i] + areas[order[1:]] - intersection, 1e-6)
        order = order[np.where(iou <= threshold)[0] + 1]
    return keep


def evaluate(outputs, image_shape, class_names, anchors, threshold, max_boxes=20):
    masks = [[3, 4, 5], [0, 1, 2]]
    input_shape = np.shape(outputs[0])[1:3] * np.asarray([32, 32])
    pairs = [_boxes_and_scores(out, anchors[masks[i]], len(class_names), input_shape, image_shape)
             for i, out in enumerate(outputs)]
    boxes = np.concatenate([p[0] for p in pairs])
    scores = np.concatenate([p[1] for p in pairs])
    final_boxes, final_scores, final_classes = [], [], []
    for class_index in range(len(class_names)):
        mask = scores[:, class_index] >= threshold
        class_boxes, class_scores = boxes[mask], scores[mask, class_index]
        keep = _nms(class_boxes, class_scores)[:max_boxes]
        final_boxes.append(class_boxes[keep])
        final_scores.append(class_scores[keep])
        final_classes.append(np.full(len(keep), class_index, dtype=np.int32))
    return tuple(np.concatenate(items) for items in (final_boxes, final_scores, final_classes))
