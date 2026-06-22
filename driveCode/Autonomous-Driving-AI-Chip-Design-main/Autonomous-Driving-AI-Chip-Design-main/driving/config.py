# Copyright (c) 2024 Sungkyunkwan University AutomationLab
#
# Authors:
# - Gyuhyeon Hwang <rbgus7080@g.skku.edu>, Hobin Oh <hobin0676@daum.net>, Minkwan Choi <arbong97@naver.com>, Hyeonjin Sim <nufxwms@naver.com>
# - url: https://micro.skku.ac.kr/micro/index.do

import numpy as np

MOTOR_ADDRESSES = {
    'motor_0': 0x00A0000000,
    'motor_1': 0x00A0010000,
    'motor_2': 0x00A0020000,
    'motor_3': 0x00A0030000,
    'motor_4': 0x00A0040000,
    'motor_5': 0x00A0050000
}

ADDRESS_RANGE = 0x10000

# YOLO configurations
anchor_list = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]
anchors = np.array(anchor_list).reshape(-1, 2)
classes_path = "../xmodel/lane_class.txt"


