import threading
import numpy as np
import spidev
from config import *


class MotorController:
    def __init__(self, motors):
        self.motors, self.lock = motors, threading.Lock()
        self.left_speed = self.right_speed = self.steering_angle = 0.0
        self.applied_left_speed = self.applied_right_speed = 0.0
        self.previous_mapped, self.velocity_ema = None, 0.0
        self.last_status = {"mapped": 0.0, "target": 0.0, "adc": 0, "cmd": "stay", "duty": 0.0}
        self.spi = spidev.SpiDev(); self.spi.open(0, 0)
        self.spi.max_speed_hz, self.spi.mode = 20_000_000, 0
        for motor in motors.values():
            motor.write(0x00, PWM_SIZE); motor.write(0x04, 0); motor.write(0x08, 0)

    def set_command(self, left, right, steering, immediate_speed=False):
        with self.lock:
            self.left_speed, self.right_speed = float(np.clip(left, -100, 100)), float(np.clip(right, -100, 100))
            self.steering_angle = float(np.clip(steering, -20, 20))
            if immediate_speed: self.applied_left_speed, self.applied_right_speed = self.left_speed, self.right_speed

    @staticmethod
    def _ramp(current, target):
        target = 0.0 if current * target < 0 else target
        rate = SPEED_ACCEL_PER_SECOND if abs(target) > abs(current) else SPEED_DECEL_PER_SECOND
        step, delta = rate / CONTROL_HZ, target-current
        return target if abs(delta) <= step else current + np.sign(delta)*step

    def _drive(self, speed, forward, reverse):
        duty = int(PWM_SIZE*abs(speed)/100)
        self.motors[forward].write(0x04, duty); self.motors[reverse].write(0x04, duty)
        self.motors[reverse].write(0x08, 0 if speed > 0 else 1)
        self.motors[forward].write(0x08, 1 if speed > 0 else 0)

    def _steer_stop(self):
        for name in ("motor_4", "motor_5"):
            self.motors[name].write(0x08, 0); self.motors[name].write(0x04, 0)

    def _steer(self, right, duty):
        active, inactive = (("motor_5", "motor_4") if right else ("motor_4", "motor_5"))
        self.motors[inactive].write(0x08, 0)
        self.motors[active].write(0x04, int(PWM_SIZE*duty)); self.motors[active].write(0x08, 1)

    def control_once(self):
        with self.lock:
            steering = self.steering_angle
            self.applied_left_speed = self._ramp(self.applied_left_speed, self.left_speed)
            self.applied_right_speed = self._ramp(self.applied_right_speed, self.right_speed)
            left, right = self.applied_left_speed, self.applied_right_speed
        raw = self.spi.xfer2([0, 0]); adc = ((raw[0]&15)<<8)|raw[1]
        mapped = 20.0 if adc <= 1294 else (-20.0 if adc >= 1883 else (1883-adc)*40/589-20)
        target, error = steering+STEER_TRIM, steering+STEER_TRIM-mapped
        if abs(error) < STEER_DEADZONE:
            self._steer_stop(); cmd, duty = "stay", 0.0
        else:
            velocity = 0.0 if self.previous_mapped is None else mapped-self.previous_mapped
            self.velocity_ema = STEER_D_FILTER*velocity+(1-STEER_D_FILTER)*self.velocity_ema
            duty = STEER_MIN_DUTY+STEER_KP*abs(error)
            if error*self.velocity_ema > 0: duty -= STEER_KD*abs(self.velocity_ema)
            duty = float(np.clip(duty, STEER_MIN_DUTY, STEER_MAX_DUTY)); cmd = "right" if error > 0 else "left"
            self._steer(error > 0, duty)
        self.previous_mapped = mapped
        self._drive(left, "motor_2", "motor_3"); self._drive(right, "motor_0", "motor_1")
        self.last_status = {"mapped": mapped, "target": target, "adc": adc, "cmd": cmd, "duty": duty}

    def drive_stopped(self): return abs(self.applied_left_speed) < 1 and abs(self.applied_right_speed) < 1

    def emergency_stop(self):
        with self.lock:
            self.left_speed = self.right_speed = self.steering_angle = 0.0
            self.applied_left_speed = self.applied_right_speed = 0.0
        self._drive(0, "motor_2", "motor_3"); self._drive(0, "motor_0", "motor_1"); self._steer_stop()

    def close(self): self.emergency_stop(); self.spi.close()
