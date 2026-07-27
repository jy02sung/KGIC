import math
import time

import cv2
import numpy as np

from config import (
    BEV_WORK_HEIGHT, CUTTING_RATIO, LANE_MAX_SATURATION, LANE_MAX_SHIFT,
    SRC_RATIO, USE_COLOR_FILTER, YOLO_THRESHOLD,
)
from yolo_utils import evaluate, pre_process


class ImageProcessor:
    def __init__(self, dpu, classes_path, anchors):
        with open(classes_path, encoding="utf-8") as class_file:
            self.class_names = [name.strip() for name in class_file if name.strip()]
        self.dpu = dpu
        self.anchors = anchors
        self.reference_point_y = 240
        self.point_detection_height = 20
        self.last_lane_center = None
        self.last_exec_ms = 0.0
        self._init_dpu_buffers()

    def _init_dpu_buffers(self):
        input_tensors = self.dpu.get_input_tensors()
        output_tensors = self.dpu.get_output_tensors()
        self.shape_in = tuple(input_tensors[0].dims)
        self.shape_out0 = tuple(output_tensors[0].dims)
        self.shape_out1 = tuple(output_tensors[1].dims)
        self.input_data = [np.empty(self.shape_in, dtype=np.float32, order="C")]
        self.output_data = [
            np.empty(self.shape_out0, dtype=np.float32, order="C"),
            np.empty(self.shape_out1, dtype=np.float32, order="C"),
        ]

    @staticmethod
    def _bird_convert(image, source, destination):
        matrix = cv2.getPerspectiveTransform(np.float32(source), np.float32(destination))
        return cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    @staticmethod
    def _calculate_angle(x1, y1, x2, y2):
        return math.degrees(math.atan2(x2 - x1, y1 - y2))

    def _is_lane_colored(self, image, box):
        if not USE_COLOR_FILTER:
            return True
        y1, x1, y2, x2 = [int(value) for value in box]
        height, width = image.shape[:2]
        y1, y2 = max(0, y1), min(height, y2)
        x1, x2 = max(0, x1), min(width, x2)
        if y2 <= y1 or x2 <= x1:
            return True
        saturation = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)[:, :, 1].mean()
        return float(saturation) <= LANE_MAX_SATURATION

    def _detect_lane_center(self, boxes, image):
        candidates = []
        for box in boxes:
            if not self._is_lane_colored(image, box):
                continue
            _, x1, _, x2 = box
            candidates.append((box, (int(x1) + int(x2)) // 2))
        if not candidates:
            return None
        if self.last_lane_center is not None:
            nearby = [item for item in candidates if abs(item[1] - self.last_lane_center) <= LANE_MAX_SHIFT]
            if not nearby:
                return None
            return min(nearby, key=lambda item: abs(item[1] - self.last_lane_center))[1]
        return max(candidates, key=lambda item: item[0][1])[1]

    @staticmethod
    def _bev_geometry(frame):
        height, width = frame.shape[:2]
        work_width = round(width * BEV_WORK_HEIGHT / height)
        work = cv2.resize(frame, (work_width, BEV_WORK_HEIGHT))
        source = [[round(x * work_width), round(y * BEV_WORK_HEIGHT)] for x, y in SRC_RATIO]
        destination = [
            [round(work_width * 0.3), 0], [round(work_width * 0.7), 0],
            [round(work_width * 0.7), BEV_WORK_HEIGHT], [round(work_width * 0.3), BEV_WORK_HEIGHT],
        ]
        return work, source, destination, round(BEV_WORK_HEIGHT * CUTTING_RATIO)

    def process_frame(self, frame):
        work, source, destination, cutting_index = self._bev_geometry(frame)
        bird = self._bird_convert(work, source, destination)
        resized = cv2.resize(bird[cutting_index:, :], (256, 256))
        image_data = pre_process(resized, (256, 256)).astype(np.float32)
        self.input_data[0][...] = image_data.reshape(self.shape_in[1:])
        start = time.time()
        job_id = self.dpu.execute_async(self.input_data, self.output_data)
        self.dpu.wait(job_id)
        self.last_exec_ms = (time.time() - start) * 1000.0
        outputs = [
            np.reshape(self.output_data[0], self.shape_out0),
            np.reshape(self.output_data[1], self.shape_out1),
        ]
        boxes, _, _ = evaluate(outputs, resized.shape[:2], self.class_names, self.anchors, YOLO_THRESHOLD)
        center = self._detect_lane_center(boxes, resized)
        self.last_lane_center = center
        if center is None:
            return None
        return int(center)
