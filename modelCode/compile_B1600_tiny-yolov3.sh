#!/bin/bash

vai_c_tensorflow2 -m quantized_yolov3_tiny/quantized_yolov3_tiny.h5 \
        -a arch.json \
        -o output/output_B1600_tiny-YOLOv3 \
        -n tiny-yolov3_256 \
        --options "{'mode':'normal','save_kernel':'', 'input_shape':'1,256,256,3'}"
