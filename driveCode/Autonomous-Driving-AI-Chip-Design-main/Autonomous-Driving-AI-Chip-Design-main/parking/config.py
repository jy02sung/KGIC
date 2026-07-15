BASE_DIR = "/home/xilinx/jupyter_notebooks/KGIC"
OVERLAY_BIT_PATH = f"{BASE_DIR}/driveCode/dpu/dpu.bit"

MOTOR_ADDRESSES = {
    "motor_0": 0x00A0000000,
    "motor_1": 0x00A0010000,
    "motor_2": 0x00A0020000,
    "motor_3": 0x00A0030000,
    "motor_4": 0x00A0040000,
    "motor_5": 0x00A0050000,
}
ADDRESS_RANGE = 0x10000
PWM_SIZE = 600600

SONIC_IP_NAME = "sonic_mm_0"
SENSOR_REGISTER_OFFSETS = [0x0, 0x4, 0x8, 0xC, 0x10]

# Sensor order follows the RTL/MMIO register order used by test_sonic.ipynb.
SENSOR_LABELS = [
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
]

# Physical pin notes from the current XDC mapping.
ULTRASONIC_WIRING = {
    "sensor_1": {"trig": "F6", "echo": "G5", "pmod": "J1[7:8]"},
    "sensor_2": {"trig": "A6", "echo": "A7", "pmod": "J1[9:10]"},
    "sensor_3": {"trig": "E5", "echo": "D6", "pmod": "J2[3:4]"},
    "sensor_4": {"trig": "D5", "echo": "C7", "pmod": "J2[7:8]"},
    "sensor_5": {"trig": "B6", "echo": "C5", "pmod": "J2[9:10]"},
}

CONTROL_HZ = 100
SENSOR_POLL_HZ = 20
DISPLAY_EVERY = 5

# These thresholds are interpreted in the same units returned by the MMIO IP.
# If the ultrasonic IP already reports centimeters, the defaults work as cm.
PARK_CRUISE_SPEED = 28
PARK_APPROACH_SPEED = 18
PARK_REVERSE_SPEED = -18
PARK_STOP_DISTANCE = 18
PARK_SLOW_DISTANCE = 35
PARK_REVERSE_DISTANCE = 10
PARK_SIDE_CLEARANCE = 14
PARK_CENTER_TARGET = 0.0
PARK_STEER_GAIN = 0.8
PARK_STEER_LIMIT = 18.0

STEER_KP = 0.026
STEER_KD = 0.035
STEER_MIN_DUTY = 0.52
STEER_MAX_DUTY = 0.80
STEER_DEADZONE = 1.2
STEER_D_FILTER = 0.35
STEER_TRIM = -5.5

MANUAL_DRIVE_SPEED = 28
MANUAL_STEER_TARGET = 16.0
