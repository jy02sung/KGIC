# Copyright (c) 2024 Sungkyunkwan University AutomationLab
#
# Authors:
# - Gyuhyeon Hwang <rbgus7080@g.skku.edu>, Hobin Oh <hobin0676@daum.net>, Minkwan Choi <arbong97@naver.com>, Hyeonjin Sim <nufxwms@naver.com>
# - url: https://micro.skku.ac.kr/micro/index.do

from pynq import Overlay, MMIO, PL, allocate
from pynq.lib.video import *
from pynq_dpu import DpuOverlay
import cv2
import numpy as np
import time
import os
import spidev
import keyboard
from driving_system_controller import DrivingSystemController
from image_processor import ImageProcessor
from config import MOTOR_ADDRESSES, ADDRESS_RANGE
from AutoLab_lib import init


init()
# Initialize SPI
spi0 = spidev.SpiDev()
spi0.open(0, 0)
spi0.max_speed_hz = 20000000
spi0.mode = 0b00

# 자율주행 모드 뒷바퀴 & 조향 속도 설정 (0 ~ 100)
speed = 50
steering_speed = 50
motors = {}

for name, addr in MOTOR_ADDRESSES.items():
    motors[name] = MMIO(addr, ADDRESS_RANGE)


def load_dpu():
    global dpu, input_data, output_data, shapeIn, shapeOut0, shapeOut1
    
    overlay = DpuOverlay("../dpu/dpu.bit")
    overlay.load_model("../xmodel/top-tiny-yolov3_coco_256.xmodel")
    
    dpu = overlay.runner
    
    return overlay, dpu

def main():
    overlay = load_dpu()
    controller = DrivingSystemController(overlay, dpu, motors, speed, steering_speed)
    controller.run(camera_index=0)

if __name__ == "__main__":
    main()