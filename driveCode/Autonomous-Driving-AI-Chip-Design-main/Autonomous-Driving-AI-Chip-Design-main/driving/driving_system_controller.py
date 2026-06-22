# Copyright (c) 2024 Sungkyunkwan University AutomationLab
#
# Authors:
# - Gyuhyeon Hwang <rbgus7080@g.skku.edu>, Hobin Oh <hobin0676@daum.net>, Minkwan Choi <arbong97@naver.com>, Hyeonjin Sim <nufxwms@naver.com>
# - url: https://micro.skku.ac.kr/micro/index.do

import cv2
import time
import keyboard
from threading import Lock

from image_processor import ImageProcessor
from motor_controller import MotorController
from config import classes_path, anchors 



class DrivingSystemController:
    def __init__(self, dpu_overlay, dpu, motors, speed, steering_speed):
        """
        자율주행 차량 시스템 초기화
        Args:
            dpu_overlay: DPU 오버레이 객체
        """
        self.image_processor = ImageProcessor(dpu, classes_path, anchors)
        self.motor_controller = MotorController(motors)
        self.overlay = dpu_overlay
        
        # 제어 상태 변수
        self.is_running = False
        self.control_lock = Lock()
        self.control_mode = 1  # 1: Autonomous, 2: Manual
        
        self.speed = speed
        self.steering_speed = steering_speed
        
        # 시스템 초기화
        self.init_system()
        
    def init_system(self):
        """시스템 초기화"""
        self.motor_controller.init_motors()

    def start_driving(self):
        """주행 시작"""
        with self.control_lock:
            self.is_running = True
            print("주행을 시작합니다.")
            if self.control_mode == 1:
                # 자율주행 모드 초기 설정
                self.motor_controller.left_speed = self.speed
                self.motor_controller.right_speed = self.speed
                self.motor_controller.steering_speed = self.steering_speed
            else:
                # 수동 주행 모드 초기 설정
                self.motor_controller.manual_speed = 0
                self.motor_controller.manual_steering_angle = 0

    def stop_driving(self):
        """주행 정지"""
        with self.control_lock:
            self.is_running = False
            print("주행을 정지합니다.")
            self.motor_controller.reset_motor_values()

    def switch_mode(self, new_mode):
        """
        주행 모드 전환
        Args:
            new_mode: 1(자율주행) 또는 2(수동주행)
        """
        if self.control_mode != new_mode:
            self.control_mode = new_mode
            self.is_running = False
            self.motor_controller.reset_motor_values()
            mode_str = "자율주행" if new_mode == 1 else "수동주행"
            print(f"{mode_str} 모드로 전환되었습니다.")
            print("Space 키를 눌러 주행을 시작하세요.")

    def process_and_control(self, frame):
        """
        프레임 처리 및 차량 제어
        Args:
            frame: 처리할 비디오 프레임
        Returns:
            처리된 이미지
        """
        if self.control_mode == 1:  # Autonomous mode
            slope, image = self.image_processor.process_frame(frame)
            if self.is_running:
                self.motor_controller.control_motors(slope, control_mode=1)
            return image
        else:  # Manual mode
            if self.is_running:
                self.motor_controller.handle_manual_control()
            return frame

    def wait_for_mode_selection(self):
        """시작 시 모드 선택 대기"""
        print("\n주행 모드를 선택하세요:")
        print("1: 자율주행 모드")
        print("2: 수동주행 모드")
        
        while True:
            if keyboard.is_pressed('1'):
                self.switch_mode(1)
                break
            elif keyboard.is_pressed('2'):
                self.switch_mode(2)
                break
            time.sleep(0.1)

    def run(self, video_path=None, camera_index=0):
        """
        메인 실행 함수
        Args:
            video_path: 비디오 파일 경로 (선택)
            camera_index: 카메라 인덱스 (기본값 0)
        """
        # 카메라 또는 비디오 초기화
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return

        # 시작 시 모드 선택
        self.wait_for_mode_selection()

        # 제어 안내 출력
        print("\n키보드 제어 안내:")
        print("Space: 주행 시작/정지")
        print("1/2: 자율주행/수동주행 모드 전환")
        if self.control_mode == 2:
            print("\n수동 주행 제어:")
            print("W/S: 전진/후진")
            print("A/D: 좌회전/우회전")
            print("R: 긴급 정지")
        print("Q: 프로그램 종료\n")

        try:
            while True:
                # 키보드 입력 처리
                if keyboard.is_pressed('space'):
                    time.sleep(0.3)  # 디바운싱
                    if self.is_running:
                        self.stop_driving()
                    else:
                        self.start_driving()
                
                elif keyboard.is_pressed('1') or keyboard.is_pressed('2'):
                    prev_mode = self.control_mode
                    new_mode = 1 if keyboard.is_pressed('1') else 2
                    if prev_mode != new_mode:
                        self.switch_mode(new_mode)
                        if new_mode == 2:
                            print("\n수동 주행 제어:")
                            print("W/S: 전진/후진")
                            print("A/D: 좌회전/우회전")
                            print("R: 긴급 정지")
                    time.sleep(0.3)  # 디바운싱
                
                if keyboard.is_pressed('q'):
                    print("\n프로그램을 종료합니다.")
                    break

                # 프레임 처리
                ret, frame = cap.read()
                if not ret:
                    print("프레임을 읽을 수 없습니다.")
                    break

                # 이미지 처리 및 차량 제어
                processed_image = self.process_and_control(frame)
                
                # 상태 표시
                mode_text = "모드: " + ("자율주행" if self.control_mode == 1 else "수동주행")
                status_text = "상태: " + ("주행중" if self.is_running else "정지")
                

        except KeyboardInterrupt:
            print("\n사용자에 의해 중지되었습니다.")
        finally:
            # 리소스 정리
            cap.release()
            cv2.destroyAllWindows()
            self.stop_driving()