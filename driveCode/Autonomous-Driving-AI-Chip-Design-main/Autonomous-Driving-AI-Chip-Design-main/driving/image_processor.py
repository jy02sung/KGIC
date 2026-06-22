# Copyright (c) 2024 Sungkyunkwan University AutomationLab
#
# Authors:
# - Gyuhyeon Hwang <rbgus7080@g.skku.edu>, Hobin Oh <hobin0676@daum.net>, Minkwan Choi <arbong97@naver.com>, Hyeonjin Sim <nufxwms@naver.com>
# - url: https://micro.skku.ac.kr/micro/index.do

import cv2
import numpy as np
import math
import os
import colorsys
import random
from PIL import Image
import time
from yolo_utils import pre_process, evaluate

class ImageProcessor:
    def __init__(self, dpu, classes_path, anchors):
        # 클래스 변수로 저장
        self.dpu = dpu
            
        self.classes_path = classes_path
        self.anchors = anchors
        self.class_names = self.load_classes(classes_path)    
        
        self.reference_point_x = 200
        self.reference_point_y = 240
        self.point_detection_height = 20
        
        # DPU 초기화 상태 추적 플래그
        self.initialized = False
        self.init_dpu()
        
    def load_classes(self, classes_path):
        """Load class names from the given path"""
        with open(classes_path, 'r') as f:
            class_names = f.read().strip().split('\n')
        return class_names 
    
    def init_dpu(self):
        """DPU 초기화 - 한 번만 실행됨"""
        if self.initialized:
            print("DPU 이미 초기화됨")
            return  # 이미 초기화되었으면 바로 리턴

        print("DPU 초기화 중...")
        inputTensors = self.dpu.get_input_tensors()
        outputTensors = self.dpu.get_output_tensors()

        self.shapeIn = tuple(inputTensors[0].dims)
        self.shapeOut0 = tuple(outputTensors[0].dims)
        self.shapeOut1 = tuple(outputTensors[1].dims)

        outputSize0 = int(outputTensors[0].get_data_size() / self.shapeIn[0])
        outputSize1 = int(outputTensors[1].get_data_size() / self.shapeIn[0])

        self.input_data = [np.empty(self.shapeIn, dtype=np.float32, order="C")]
        self.output_data = [
            np.empty(self.shapeOut0, dtype=np.float32, order="C"),
            np.empty(self.shapeOut1, dtype=np.float32, order="C")
        ]

        # 초기화 완료 플래그 설정
        self.initialized = True
        print("DPU 초기화 완료")

    
    def roi_rectangle_below(self, img, cutting_idx):
        return img[cutting_idx:, :]

    def warpping(self, image, srcmat, dstmat):
        h, w = image.shape[0], image.shape[1]
        transform_matrix = cv2.getPerspectiveTransform(srcmat, dstmat)
        minv = cv2.getPerspectiveTransform(dstmat, srcmat)
        _image = cv2.warpPerspective(image, transform_matrix, (w, h))
        return _image, minv
    
    def bird_convert(self, img, srcmat, dstmat):
        srcmat = np.float32(srcmat)
        dstmat = np.float32(dstmat)
        img_warpped, _ = self.warpping(img, srcmat, dstmat)
        return img_warpped

    def calculate_angle(self, x1, y1, x2, y2):
        if x1 == x2:
            return 90.0
        slope = (y2 - y1) / (x2 - x1)
        return -math.degrees(math.atan(slope))

    def detect_lane_center_x(self, xyxy_results):
        rightmost_lane_x_min = None
        rightmost_lane_x_max = None
        rightmost_x_position = -float('inf')
        
        for box in xyxy_results:
            y1, x1, y2, x2 = box
            if x1 > rightmost_x_position:
                rightmost_x_position = x1
                rightmost_lane_x_min = int(x1)
                rightmost_lane_x_max = int(x2)
        
        if rightmost_lane_x_min is not None and rightmost_lane_x_max is not None:
            return (rightmost_lane_x_min + rightmost_lane_x_max) // 2
        return None

    def process_frame(self, img):

        
        h, w = img.shape[0], img.shape[1]
        dst_mat = [[round(w * 0.3), 0], [round(w * 0.7), 0], 
                  [round(w * 0.7), h], [round(w * 0.3), h]]
        src_mat = [[238, 316], [402, 313], [501, 476], [155, 476]]
        
        bird_img = self.bird_convert(img, srcmat=src_mat, dstmat=dst_mat)
        roi_image = self.roi_rectangle_below(bird_img, cutting_idx=300)
        
        img = cv2.resize(roi_image, (256, 256))
        image_size = img.shape[:2]
        image_data = np.array(pre_process(img, (256, 256)), dtype=np.float32)
        
        start_time = time.time()

        # self를 사용하여 클래스 변수에 접근
        self.input_data[0][...] = image_data.reshape(self.shapeIn[1:])
        job_id = self.dpu.execute_async(self.input_data, self.output_data)
        self.dpu.wait(job_id)
        end_time = time.time()
        
        conv_out0 = np.reshape(self.output_data[0], self.shapeOut0)
        conv_out1 = np.reshape(self.output_data[1], self.shapeOut1)
        yolo_outputs = [conv_out0, conv_out1]

        boxes, scores, classes = evaluate(yolo_outputs, image_size, self.class_names, self.anchors)

        for i, box in enumerate(boxes):
            top_left = (int(box[1]), int(box[0]))
            bottom_right = (int(box[3]), int(box[2]))
            #cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

        right_lane_center = self.detect_lane_center_x(boxes)
        
        ######### 예외처리 알고리즘 #########
        if right_lane_center is None:
            print("차선 중심을 찾을 수 없습니다.")
            right_lane_angle = 90
            return right_lane_angle, img
        ###################################

        right_lane_angle = self.calculate_angle(self.reference_point_x, self.reference_point_y, right_lane_center, self.point_detection_height)
        
        return right_lane_angle, img
