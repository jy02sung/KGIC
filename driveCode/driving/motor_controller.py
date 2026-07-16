import threading

import numpy as np
import spidev

from config import (
    CONTROL_HZ, PWM_SIZE, STEER_D_FILTER, STEER_DEADZONE, STEER_KD, STEER_KP,
    STEER_MAX_DUTY, STEER_MIN_DUTY, STEER_TRIM,
)


class MotorController:
    """Final vehicle motor mapping and 100 Hz steering PD loop."""

    def __init__(self, motors):
        self.motors = motors
        self.size = PWM_SIZE
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.steering_angle = 0.0
        self.resistance_most_left = 2338
        self.resistance_most_right = 1512
        self._previous_mapped = None
        self._velocity_ema = 0.0
        self._lock = threading.Lock()

        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 20000000
        self.spi.mode = 0b00
        self.init_motors()

    def init_motors(self):
        for motor in self.motors.values():
            motor.write(0x00, self.size)
            motor.write(0x04, 0)
            motor.write(0x08, 0)

    def set_command(self, left_speed, right_speed, steering_angle):
        with self._lock:
            self.left_speed = float(np.clip(left_speed, -100, 100))
            self.right_speed = float(np.clip(right_speed, -100, 100))
            self.steering_angle = float(np.clip(steering_angle, -20, 20))

    def read_adc(self):
        response = self.spi.xfer2([0x00, 0x00])
        return ((response[0] & 0x0F) << 8) | response[1]

    @staticmethod
    def map_value(value, in_min, in_max, out_min, out_max):
        if value <= in_min:
            return out_max
        if value >= in_max:
            return out_min
        return (in_max - value) * (out_max - out_min) / (in_max - in_min) + out_min

    def _right(self, duty_ratio):
        duty = int(self.size * np.clip(duty_ratio, 0.0, 1.0))
        self.motors["motor_4"].write(0x08, 0)
        self.motors["motor_5"].write(0x04, duty)
        self.motors["motor_5"].write(0x08, 1)

    def _left(self, duty_ratio):
        duty = int(self.size * np.clip(duty_ratio, 0.0, 1.0))
        self.motors["motor_5"].write(0x08, 0)
        self.motors["motor_4"].write(0x04, duty)
        self.motors["motor_4"].write(0x08, 1)

    def _stay(self):
        for name in ("motor_4", "motor_5"):
            self.motors[name].write(0x08, 0)
            self.motors[name].write(0x04, 0)

    def _pd_duty(self, error, mapped_resistance):
        raw_velocity = 0.0 if self._previous_mapped is None else mapped_resistance - self._previous_mapped
        self._previous_mapped = mapped_resistance
        self._velocity_ema = STEER_D_FILTER * raw_velocity + (1.0 - STEER_D_FILTER) * self._velocity_ema
        duty = STEER_MIN_DUTY + STEER_KP * abs(error)
        if error * self._velocity_ema > 0:
            duty -= STEER_KD * abs(self._velocity_ema)
        return float(np.clip(duty, STEER_MIN_DUTY, STEER_MAX_DUTY))

    def _set_left_speed(self, speed):
        duty = int(self.size * abs(speed) / 100.0)
        self.motors["motor_2"].write(0x04, duty)
        self.motors["motor_3"].write(0x04, duty)
        if speed > 0:
            self.motors["motor_3"].write(0x08, 0)
            self.motors["motor_2"].write(0x08, 1)
        else:
            self.motors["motor_3"].write(0x08, 1)
            self.motors["motor_2"].write(0x08, 0)

    def _set_right_speed(self, speed):
        duty = int(self.size * abs(speed) / 100.0)
        self.motors["motor_1"].write(0x04, duty)
        self.motors["motor_0"].write(0x04, duty)
        if speed > 0:
            self.motors["motor_1"].write(0x08, 0)
            self.motors["motor_0"].write(0x08, 1)
        else:
            self.motors["motor_1"].write(0x08, 1)
            self.motors["motor_0"].write(0x08, 0)

    def control_once(self):
        with self._lock:
            left_speed, right_speed, steering_angle = self.left_speed, self.right_speed, self.steering_angle
        mapped = self.map_value(
            self.read_adc(), self.resistance_most_right, self.resistance_most_left, -20, 20
        )
        error = steering_angle + STEER_TRIM - mapped
        if abs(error) < STEER_DEADZONE:
            self._stay()
            command, duty = "stay", 0.0
        else:
            duty = self._pd_duty(error, mapped)
            if error < 0:
                self._left(duty)
                command = "left"
            else:
                self._right(duty)
                command = "right"
        self._set_left_speed(left_speed)
        self._set_right_speed(right_speed)
        return {"mapped": mapped, "target": steering_angle + STEER_TRIM, "error": error, "cmd": command, "duty": duty}

    def stop(self):
        self.set_command(0, 0, 0)
        self._stay()
        self._set_left_speed(0)
        self._set_right_speed(0)

    def close(self):
        self.stop()
        self.spi.close()
