import threading
import time

import cv2
import keyboard
import numpy as np

from config import (
    CAPTURE_H, CAPTURE_W, CENTER_EMA, CONTROL_HZ, DRIVE_SPEED, HEARTBEAT_EVERY,
    LOST_HOLD_FRAMES, MANUAL_DRIVE_SPEED, MANUAL_STEER_TARGET, RACING_DRIVE_SPEED,
    REF_X, STEER_DIR, STEER_GAIN,
)
from racing_line_controller import RacingLineController


class LatestFrameGrabber:
    """Autonomous/racing capture thread that discards stale frames."""

    def __init__(self, capture):
        self.capture = capture
        self._lock = threading.Lock()
        self._frame = None
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame

    def read_latest(self, timeout=1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None:
                    frame = self._frame
                    self._frame = None
                    return True, frame
            time.sleep(0.001)
        return False, None

    def stop(self):
        self._stop = True
        self._thread.join(timeout=5.0)
        return not self._thread.is_alive()


class DrivingSystemController:
    """Board runner: 1 autonomous, 2 manual, 3 continuous racing line."""

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
        self.racing_line = RacingLineController()
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)

    def _control_loop(self):
        period = 1.0 / CONTROL_HZ
        previous = time.monotonic()
        while not self.stop_requested:
            feedback = self.motor_controller.control_once()
            now = time.monotonic()
            dt = now - previous
            previous = now
            if self.mode == "race" and self.running:
                # Track integration remains at 100 Hz. The vision loop only does
                # an O(1) reference lookup, so racing mode adds no frame latency.
                self.racing_line.update(
                    feedback["mapped"] - feedback["trim"],
                    dt,
                    commanded_steer=self._last_steering,
                )
            time.sleep(period)

    def _pressed_once(self, key):
        is_down = keyboard.is_pressed(key)
        was_down = self._last_key_state.get(key, False)
        self._last_key_state[key] = is_down
        return is_down and not was_down

    @staticmethod
    def _vision_to_target(center):
        return float(np.clip(STEER_DIR * (center - REF_X) * STEER_GAIN, -20, 20))

    @staticmethod
    def _vision_to_racing_target(center, ref_x):
        return float(np.clip(STEER_DIR * (center - ref_x) * STEER_GAIN, -20, 20))

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

    def _update_racing_command(self, center, frame_dt):
        if center is None:
            self._lost_frames += 1
        else:
            self._lost_frames = 0

        lane_ok = center is not None or self._lost_frames <= LOST_HOLD_FRAMES
        ref_x = self.racing_line.compute_ref_x(frame_dt, lane_ok=lane_ok)
        state = self.racing_line.snapshot()

        if center is None:
            if self._lost_frames > LOST_HOLD_FRAMES:
                self._last_steering = 0.0
                self._center_ema = None
        else:
            if self._center_ema is None:
                self._center_ema = float(center)
            else:
                self._center_ema = CENTER_EMA * center + (1.0 - CENTER_EMA) * self._center_ema
            self._last_steering = self._vision_to_racing_target(self._center_ema, ref_x)
        self.motor_controller.set_command(
            RACING_DRIVE_SPEED, RACING_DRIVE_SPEED, self._last_steering
        )
        return state

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
        if self._pressed_once("3"):
            self.running = False
            self.mode = "race"
            self.racing_line.reset()
            self._center_ema = None
            self._last_steering = 0.0
            self._lost_frames = 0
            self._set_stopped_command()
            print("Mode: racing line (continuous laps)")
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
        print("Keys: 1 auto, 2 manual, 3 racing line (continuous), Space start/stop, W/A/S/D manual, Q quit")
        frame_index = 0
        previous_frame_time = time.monotonic()
        frame_grabber = None
        capture_can_release = True
        try:
            while not self.stop_requested:
                self._handle_mode_keys()

                if self.mode in ("auto", "race") and frame_grabber is None:
                    frame_grabber = LatestFrameGrabber(cap)
                elif self.mode == "manual" and frame_grabber is not None:
                    if not frame_grabber.stop():
                        capture_can_release = False
                        raise RuntimeError("Capture thread did not stop; restart the process")
                    frame_grabber = None

                if self.mode == "manual":
                    previous_frame_time = time.monotonic()
                    if self.running:
                        self._update_manual_command()
                    else:
                        self._set_stopped_command()
                    time.sleep(0.005)
                    continue

                if frame_grabber is not None:
                    ret, frame = frame_grabber.read_latest()
                else:
                    ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Camera frame read failed")
                if self.running:
                    frame_now = time.monotonic()
                    frame_dt = frame_now - previous_frame_time
                    previous_frame_time = frame_now
                    # Mode 1 keeps the safer resize but combines crop/warp;
                    # mode 3 uses notebook-v15's fastest direct single warp.
                    bev_mode = "fastest" if self.mode == "race" else "fast"
                    center = self.image_processor.process_frame(frame, bev_mode=bev_mode)
                    if self.mode == "race":
                        race_state = self._update_racing_command(center, frame_dt)
                    else:
                        self._update_auto_command(center)
                    if frame_index % HEARTBEAT_EVERY == 0:
                        if self.mode == "race":
                            print(
                                "[RACE] lap={} seg={} frame={} center={} ref={:.1f} "
                                "lost={} steer={:+.1f} dpu={:.1f}ms".format(
                                    race_state["lap"], race_state["segment"], frame_index,
                                    center, race_state["ref_x"], self._lost_frames,
                                    self._last_steering, self.image_processor.last_exec_ms,
                                )
                            )
                        else:
                            print("[AUTO] frame={} center={} lost={} steer={:+.1f} dpu={:.1f}ms".format(
                                frame_index, center, self._lost_frames, self._last_steering,
                                self.image_processor.last_exec_ms,
                            ))
                    frame_index += 1
                else:
                    self._set_stopped_command()
                    previous_frame_time = time.monotonic()
                    time.sleep(0.01)
        finally:
            self.stop_requested = True
            self._control_thread.join(timeout=1.0)
            if frame_grabber is not None and not frame_grabber.stop():
                capture_can_release = False
            # Releasing VideoCapture while cap.read() is still running can crash
            # OpenCV and the board. Leak it and require a process restart instead.
            if capture_can_release:
                cap.release()
            self.motor_controller.close()
