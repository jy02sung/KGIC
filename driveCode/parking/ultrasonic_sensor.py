from config import (RIGHT_FRONT_SENSOR_OFFSET, RIGHT_REAR_SENSOR_OFFSET,
                    ULTRASONIC_CLOCK_HZ, ULTRASONIC_FAR_DISTANCE_CM,
                    ULTRASONIC_INVALID_DISTANCE_CM, ULTRASONIC_MAX_DISTANCE_CM)


class UltrasonicSensorPair:
    def __init__(self, mmio): self.mmio = mmio

    def read_cm(self, offset):
        ticks = int(self.mmio.read(offset))
        if ticks <= 0: return ULTRASONIC_INVALID_DISTANCE_CM
        distance = ticks * 34300.0 / (2.0 * ULTRASONIC_CLOCK_HZ)
        return ULTRASONIC_FAR_DISTANCE_CM if distance >= ULTRASONIC_MAX_DISTANCE_CM else distance

    def read_pair(self):
        return self.read_cm(RIGHT_FRONT_SENSOR_OFFSET), self.read_cm(RIGHT_REAR_SENSOR_OFFSET)
