import threading
import time

import keyboard
import numpy as np

from config import (
    CONTROL_HZ, DISPLAY_EVERY, MANUAL_DRIVE_SPEED, MANUAL_STEER_TARGET,
    PARK_APPROACH_SPEED, PARK_CENTER_TARGET, PARK_CRUISE_SPEED,
    PARK_REVERSE_DISTANCE, PARK_REVERSE_SPEED, PARK_SIDE_CLEARANCE,
    PARK_SLOW_DISTANCE, PARK_STEER_GAIN, PARK_STEER_LIMIT, PARK_STOP_DISTANCE,
    SENSOR_POLL_HZ,
)


class ParkingSystemController:
    """Ultrasonic parking assist with the same keyboard flow as driving mode."""

    def __init__(self, ultrasonic_sensor, motor_controller):
        self.ultrasonic_sensor = ultrasonic_sensor
        self.motor_controller = motor_controller
        self.mode = "assist"
        self.running = False
        self.stop_requested = False
        self._last_key_state = {}
        self._frame_index = 0
        self._latest_summary = None
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

    def _set_stopped_command(self):
        self.motor_controller.set_command(0, 0, 0)

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

    def _compute_assist_command(self, summary):
        center = summary["center"]
        left_avg = summary["left_avg"]
        right_avg = summary["right_avg"]
        minimum = summary["minimum"]

        if minimum <= PARK_REVERSE_DISTANCE:
            return PARK_REVERSE_SPEED, PARK_REVERSE_SPEED, 0.0, "reverse_escape"
        if center <= PARK_STOP_DISTANCE:
            return 0.0, 0.0, 0.0, "stop_center"

        if center <= PARK_SLOW_DISTANCE:
            speed = PARK_APPROACH_SPEED
        else:
            speed = PARK_CRUISE_SPEED

        if left_avg <= PARK_SIDE_CLEARANCE and right_avg <= PARK_SIDE_CLEARANCE:
            steering = 0.0
            state = "corridor"
        else:
            balance = (left_avg - right_avg) - PARK_CENTER_TARGET
            steering = float(np.clip(balance * PARK_STEER_GAIN, -PARK_STEER_LIMIT, PARK_STEER_LIMIT))
            state = "approach"
        return speed, speed, steering, state

    def _handle_mode_keys(self):
        if self._pressed_once("1"):
            self.mode = "assist"
            self.running = False
            self._set_stopped_command()
            print("Mode: parking assist")
        if self._pressed_once("2"):
            self.mode = "manual"
            self.running = False
            self._set_stopped_command()
            print("Mode: manual")
        if self._pressed_once("space"):
            self.running = not self.running
            if not self.running:
                self._set_stopped_command()
            print("Parking: {}".format("start" if self.running else "stop"))
        if self._pressed_once("q"):
            self.stop_requested = True

    def run(self):
        self._control_thread.start()
        print("Keys: 1 assist, 2 manual, Space start/stop, W/A/S/D manual, Q quit")
        period = 1.0 / SENSOR_POLL_HZ
        try:
            while not self.stop_requested:
                self._handle_mode_keys()
                summary = self.ultrasonic_sensor.read_summary()
                self._latest_summary = summary

                if self.mode == "manual":
                    if self.running:
                        self._update_manual_command()
                    else:
                        self._set_stopped_command()
                else:
                    if self.running:
                        left_speed, right_speed, steering, state = self._compute_assist_command(summary)
                        self.motor_controller.set_command(left_speed, right_speed, steering)
                    else:
                        steering = 0.0
                        state = "idle"
                        self._set_stopped_command()

                if self._frame_index % DISPLAY_EVERY == 0:
                    values = " ".join(f"{key}={value}" for key, value in summary["values"].items())
                    print(
                        "[{}] {} center={} left_avg={:.1f} right_avg={:.1f} min={} steer={:+.1f}".format(
                            self.mode.upper(),
                            state,
                            summary["center"],
                            summary["left_avg"],
                            summary["right_avg"],
                            summary["minimum"],
                            steering,
                        )
                    )
                    print("    " + values)
                self._frame_index += 1
                time.sleep(period)
        finally:
            self.stop_requested = True
            self._control_thread.join(timeout=1.0)
            self.motor_controller.close()
