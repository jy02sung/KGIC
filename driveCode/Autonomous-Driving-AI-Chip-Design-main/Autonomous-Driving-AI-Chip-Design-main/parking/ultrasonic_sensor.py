from collections import OrderedDict

from config import SENSOR_LABELS, SENSOR_REGISTER_OFFSETS


class UltrasonicSensorArray:
    """MMIO-backed 5-channel ultrasonic sensor reader."""

    def __init__(self, mmio, labels=None, offsets=None):
        self.mmio = mmio
        self.labels = list(labels or SENSOR_LABELS)
        self.offsets = list(offsets or SENSOR_REGISTER_OFFSETS)
        if len(self.labels) != len(self.offsets):
            raise ValueError("Sensor labels and offsets must have the same length")

    def read_raw(self):
        values = OrderedDict()
        for label, offset in zip(self.labels, self.offsets):
            values[label] = int(self.mmio.read(offset))
        return values

    def read_summary(self):
        values = self.read_raw()
        ordered = list(values.values())
        center = ordered[len(ordered) // 2]
        left_avg = sum(ordered[:2]) / 2.0
        right_avg = sum(ordered[-2:]) / 2.0
        return {
            "values": values,
            "center": center,
            "left_avg": left_avg,
            "right_avg": right_avg,
            "minimum": min(ordered),
        }
