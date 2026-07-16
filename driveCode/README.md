# Autonomous Driving AI Chip Design Baseline File Structure

2026 Autonomous Driving AI Chip Design for Sungkyunkwan University Students

---

## Related Repositories & Resources

- [YOLACT_SKKU](https://github.com/SKKUAutoLab/YOLACT_SKKU)  
- [yolact_vitis_ai_SKKU](https://github.com/SKKUAutoLab/yolact_vitis_ai_SKKU)  
- [Tiny-yolov3-keras](https://github.com/SKKUAutoLab/Tiny-yolov3-keras)  
- [ros2_autonomous_vehicle_simulation](https://github.com/SKKUAutoLab/ros2_autonomous_vehicle_simulation)  
- [PWM RTL Code (Google Drive)](https://drive.google.com/file/d/1zA4lm_GMxx4Rb-orluPuT32Hf5BqDOJo/view?usp=drive_link)  
- [Test dataset (Google Drive)](https://drive.google.com/file/d/1dNZBiT1rwPjCdPhqsQOfeQndNe5JSUiN/view?usp=drive_link)  
- [YOLOv3-tiny Vitis-ai (Google Drive)](https://drive.google.com/file/d/16zhGDqBl_MWdsYOjycSokO4XoY-AF_yK/view?usp=drive_link)  
- [Educational Material (Google Drive)](https://drive.google.com/file/d/1AzIVBAaPTiS8fYsKFQ7jMrLIPk1s4KEZ/view?usp=drive_link)  

---

## Folder Structure

### $\textcolor{red}{\mathbf{(NEW)}}$ Segmentation

- **test_data**  
  *Inference test images*
- **dpu_yolact.ipynb**  
  *Jupyter Notebook for real-time segmentation inference (~20FPS)*

### debugging
- **SoC_Driving.ipynb**  
  *Jupyter Notebook for driving code*
- **data_collection.ipynb**  
  *Jupyter Notebook for data collection*
- $\textcolor{red}{\mathbf{(NEW)}}$ **test_sonic.ipynb**  
  *Jupyter Notebook for Ultrasonic sensor test (for reference only, simple example)*

### dpu
- **dpu.bit**  
  *DPU bitstream file **(Students need to add)***
- **dpu.hwh**  
  *DPU hardware file **(Students need to add)***
- **dpu.xclbin**  
  *DPU executable file **(Students need to add)***

### driving
- **config.py**  
  *Initial motor address settings*
- **driving_system_controller.py**  
  *Driving mode settings*
- **image_processor.py**  
  *Image processing script*
- **main.py**  
  *Driving parameter settings*
- **motor_controller.py**  
  *Motor control settings*
- **yolo_utils.py**  
  *YOLO utility functions*

### test_video
- **test_video.mp4**  
  *Test video file*

### xmodel
- **lane_class.txt**  
  *Model class configuration **(Students need to add)***
- **top-tiny-yolov3_coco_256.xmodel**  
  *Compiled deep learning model file **(Students need to add)***

---
