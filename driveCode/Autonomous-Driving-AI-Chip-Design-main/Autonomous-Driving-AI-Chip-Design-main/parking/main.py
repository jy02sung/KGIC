import logging

from pynq import MMIO, Overlay

from AutoLab_lib import init
from config import ADDRESS_RANGE, MOTOR_ADDRESSES, OVERLAY_BIT_PATH, SONIC_IP_NAME
from motor_controller import MotorController
from parking_system_controller import ParkingSystemController
from ultrasonic_sensor import UltrasonicSensorArray


def create_motors():
    return {name: MMIO(address, ADDRESS_RANGE) for name, address in MOTOR_ADDRESSES.items()}


def create_ultrasonic_sensor(overlay):
    if not hasattr(overlay, SONIC_IP_NAME):
        raise AttributeError(
            "Overlay does not expose '{}'. Check the IP instance name in the exported design.".format(SONIC_IP_NAME)
        )
    return UltrasonicSensorArray(getattr(overlay, SONIC_IP_NAME).mmio)


def main():
    logging.basicConfig(level=logging.INFO)
    init()
    overlay = Overlay(OVERLAY_BIT_PATH)
    ultrasonic_sensor = create_ultrasonic_sensor(overlay)
    motor_controller = MotorController(create_motors())
    ParkingSystemController(ultrasonic_sensor, motor_controller).run()


if __name__ == "__main__":
    main()
