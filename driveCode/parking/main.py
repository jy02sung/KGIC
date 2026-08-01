import logging
from pynq import MMIO
from pynq_dpu import DpuOverlay
from config import (ADDRESS_RANGE, ANCHORS, CLASSES_PATH, DPU_BIT_PATH,
                    MODEL_PATH, MOTOR_ADDRESSES, SONIC_IP_NAME)
from image_processor import ImageProcessor
from motor_controller import MotorController
from parking_system_controller import ParkingSystemController
from ultrasonic_sensor import UltrasonicSensorPair


def main():
    logging.basicConfig(level=logging.INFO)
    overlay = DpuOverlay(DPU_BIT_PATH)
    overlay.load_model(MODEL_PATH)
    if not hasattr(overlay, SONIC_IP_NAME):
        raise AttributeError(f"Overlay does not expose {SONIC_IP_NAME}")
    motors = {name: MMIO(address, ADDRESS_RANGE) for name, address in MOTOR_ADDRESSES.items()}
    image = ImageProcessor(overlay.runner, CLASSES_PATH, ANCHORS)
    sonic = UltrasonicSensorPair(getattr(overlay, SONIC_IP_NAME).mmio)
    ParkingSystemController(image, sonic, MotorController(motors)).run(0)


if __name__ == "__main__": main()
