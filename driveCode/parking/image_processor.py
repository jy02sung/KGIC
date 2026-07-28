import time

import cv2
import numpy as np

from config import (BEV_WORK_HEIGHT, CUTTING_RATIO, LANE_MAX_SATURATION,
                    LANE_MAX_SHIFT, REF_X, SRC_RATIO, USE_COLOR_FILTER,
                    YOLO_THRESHOLD)
from yolo_utils import evaluate, pre_process


class ImageProcessor:
    def __init__(self, dpu, classes_path, anchors):
        with open(classes_path, encoding="utf-8") as class_file:
            self.class_names = [line.strip() for line in class_file if line.strip()]
        self.dpu = dpu
        self.anchors = anchors
        self.lane_follow_side = "right"
        self.last_lane_center = None
        self.last_boxes = []
        self.last_scores = []
        self.last_exec_ms = 0.0
        inputs, outputs = dpu.get_input_tensors(), dpu.get_output_tensors()
        self.shape_in = tuple(inputs[0].dims)
        self.shape_out = [tuple(tensor.dims) for tensor in outputs]
        self.input_data = [np.empty(self.shape_in, dtype=np.float32, order="C")]
        self.output_data = [np.empty(shape, dtype=np.float32, order="C") for shape in self.shape_out]

    @staticmethod
    def _geometry(frame):
        h, w = frame.shape[:2]
        work_w = round(w * BEV_WORK_HEIGHT / h)
        work = cv2.resize(frame, (work_w, BEV_WORK_HEIGHT))
        source = [[round(x*work_w), round(y*BEV_WORK_HEIGHT)] for x, y in SRC_RATIO]
        destination = [[round(work_w*.3), 0], [round(work_w*.7), 0],
                       [round(work_w*.7), BEV_WORK_HEIGHT], [round(work_w*.3), BEV_WORK_HEIGHT]]
        return work, source, destination, round(BEV_WORK_HEIGHT * CUTTING_RATIO)

    @staticmethod
    def _is_lane_colored(image, box):
        if not USE_COLOR_FILTER:
            return True
        y1, x1, y2, x2 = [int(v) for v in box]
        h, w = image.shape[:2]
        y1, y2, x1, x2 = max(0, y1), min(h, y2), max(0, x1), min(w, x2)
        if y2 <= y1 or x2 <= x1:
            return True
        saturation = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)[:, :, 1].mean()
        return float(saturation) <= LANE_MAX_SATURATION

    def _select_center(self, boxes, image):
        candidates = [(box, (int(box[1]) + int(box[3])) // 2)
                      for box in boxes if self._is_lane_colored(image, box)]
        if not candidates:
            return None
        if self.last_lane_center is not None:
            nearby = [item for item in candidates if abs(item[1]-self.last_lane_center) <= LANE_MAX_SHIFT]
            return None if not nearby else min(nearby, key=lambda item: abs(item[1]-self.last_lane_center))[1]
        if self.lane_follow_side == "left":
            return min(candidates, key=lambda item: item[0][1])[1]
        # Final notebook behavior: initial right-lane acquisition chooses rightmost box.
        return max(candidates, key=lambda item: item[0][1])[1]

    def process_frame(self, frame):
        work, source, destination, cut = self._geometry(frame)
        matrix = cv2.getPerspectiveTransform(np.float32(source), np.float32(destination))
        bird = cv2.warpPerspective(work, matrix, (work.shape[1], work.shape[0]))
        resized = cv2.resize(bird[cut:, :], (256, 256))
        self.input_data[0][...] = pre_process(resized, (256, 256)).reshape(self.shape_in)
        started = time.time()
        job = self.dpu.execute_async(self.input_data, self.output_data)
        self.dpu.wait(job)
        self.last_exec_ms = (time.time() - started) * 1000.0
        outputs = [np.reshape(data, shape) for data, shape in zip(self.output_data, self.shape_out)]
        boxes, scores, _ = evaluate(outputs, resized.shape[:2], self.class_names,
                                    self.anchors, YOLO_THRESHOLD)
        self.last_boxes = [[round(float(v), 1) for v in box] for box in boxes]
        self.last_scores = [round(float(score), 3) for score in scores]
        for box in boxes:
            cv2.rectangle(resized, (int(box[1]), int(box[0])),
                          (int(box[3]), int(box[2])), (0, 255, 0), 2)
        center = self._select_center(boxes, resized)
        self.last_lane_center = center
        if center is not None:
            cv2.line(resized, (REF_X, 240), (center, 20), (0, 0, 255), 3)
            cv2.circle(resized, (center, 20), 6, (0, 255, 255), -1)
        return center, resized
