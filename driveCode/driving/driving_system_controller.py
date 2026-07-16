import threading
import time

import cv2
import keyboard
import numpy as np

from config import (
    CAPTURE_H, CAPTURE_W, CENTER_EMA, CONTROL_HZ, DRIVE_SPEED, HEARTBEAT_EVERY,
    LOST_HOLD_FRAMES, MANUAL_DRIVE_SPEED, MANUAL_STEER_TARGET, REF_X, STEER_DIR,
    STEER_GAIN,
)


class DrivingSystemController:
    """Board-only autonomous/manual runner controlled by a USB wireless keyboard."""

    def __init__(self, image_processor, motor_controller):
        self.image_processor = image_processor
        self.motor_controller = motor_controller
        self.mode = "auto"
        self.running = False
        self.stop_requested = False
        self._center_ema = None
        self._last_steering = 0.0
        self._lost_frames = 0
        self._last_key_state = {}
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)

    def _control_loop(self):
        period = 1.0 / CONTROL_HZ
        while not self.stop_requested:
            self.motor_controller.control_once()
            time.sleep(period)

    def _pressed_once(self, key):
        is_down = keyboard.is_pressed(key)
        was_down = self._last_key_state.get(key, False)
        self._last_key_state[key] = is_down
        return is_down and not was_down

    @staticmethod
    def _vision_to_target(center):
        return float(np.clip(STEER_DIR * (center - REF_X) * STEER_GAIN, -20, 20))

    def _set_stopped_command(self):
        self.motor_controller.set_command(0, 0, 0)

    def _update_auto_command(self, center):
        if center is None:
            self._lost_frames += 1
            if self._lost_frames > LOST_HOLD_FRAMES:
                self._last_steering = 0.0
                self._center_ema = None
        else:
            self._lost_frames = 0
            if self._center_ema is None:
                self._center_ema = float(center)
            else:
                self._center_ema = CENTER_EMA * center + (1.0 - CENTER_EMA) * self._center_ema
            self._last_steering = self._vision_to_target(self._center_ema)
        self.motor_controller.set_command(DRIVE_SPEED, DRIVE_SPEED, self._last_steering)

    def _update_manual_command(self):
        if keyboard.is_pressed("w") and not keyboard.is_pressed("s"):
            speed = MANUAL_DRIVE_SPEED
        elif keyboard.is_pressed("s") and not keyboard.is_pressed("w"):
            speed = -MANUAL_DRIVE_SPEED
        else:
            speed = 0
        if keyboard.is_pressed("a") and not keyboard.is_pressed("d"):
            steering = -MANUAL_STEER_TARGET
        elif keyboard.is_pressed("d") and not keyboard.is_pressed("a"):
            steering = MANUAL_STEER_TARGET
        else:
            steering = 0
        self.motor_controller.set_command(speed, speed, steering)

    def _handle_mode_keys(self):
        if self._pressed_once("1"):
            self.running = False
            self.mode = "auto"
            self._set_stopped_command()
            print("Mode: autonomous")
        if self._pressed_once("2"):
            self.running = False
            self.mode = "manual"
            self._set_stopped_command()
            print("Mode: manual")
        if self._pressed_once("space"):
            self.running = not self.running
            if not self.running:
                self._set_stopped_command()
            print("Drive: {}".format("start" if self.running else "stop"))
        if self._pressed_once("q"):
            self.stop_requested = True

    def run(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        if not cap.isOpened():
            raise RuntimeError("Camera could not be opened")
        self._control_thread.start()
        print("Keys: 1 auto, 2 manual, Space start/stop, W/A/S/D manual, Q quit")
        frame_index = 0
        try:
            while not self.stop_requested:
                self._handle_mode_keys()
                if self.mode == "manual":
                    if self.running:
                        self._update_manual_command()
                    else:
                        self._set_stopped_command()
                    time.sleep(0.005)
                    continue

                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Camera frame read failed")
                if self.running:
                    center = self.image_processor.process_frame(frame)
                    self._update_auto_command(center)
                    if frame_index % HEARTBEAT_EVERY == 0:
                        print("[AUTO] frame={} center={} lost={} steer={:+.1f} dpu={:.1f}ms".format(
                            frame_index, center, self._lost_frames, self._last_steering,
                            self.image_processor.last_exec_ms,
                        ))
                    frame_index += 1
                else:
                    self._set_stopped_command()
                    time.sleep(0.01)
        finally:
            self.stop_requested = True
            self._control_thread.join(timeout=1.0)
            cap.release()
            self.motor_controller.close()
