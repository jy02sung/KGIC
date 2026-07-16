import logging

from pynq import MMIO
from pynq_dpu import DpuOverlay

from AutoLab_lib import init
from config import ADDRESS_RANGE, ANCHORS, CLASSES_PATH, DPU_BIT_PATH, MODEL_PATH, MOTOR_ADDRESSES
from driving_system_controller import DrivingSystemController
from image_processor import ImageProcessor
from motor_controller import MotorController


def create_motors():
    return {name: MMIO(address, ADDRESS_RANGE) for name, address in MOTOR_ADDRESSES.items()}


def main():
    logging.basicConfig(level=logging.INFO)
    init()
    overlay = DpuOverlay(DPU_BIT_PATH)
    overlay.load_model(MODEL_PATH)
    image_processor = ImageProcessor(overlay.runner, CLASSES_PATH, ANCHORS)
    motor_controller = MotorController(create_motors())
    DrivingSystemController(image_processor, motor_controller).run(camera_index=0)


if __name__ == "__main__":
    main()
