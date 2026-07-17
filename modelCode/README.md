# 🚀 YOLOv3-Tiny Vitis-AI (End-to-End Workflow)

> 💡 **The 2026 Winter Extracurricular Program – Autonomous Driving SoC Design**  
>  
> This project provides a fully integrated workflow for **Training → Quantization → Compilation** of the **YOLOv3-Tiny** model inside the Vitis-AI environment.  
>  
> **Participants must prepare their own datasets**, then follow this pipeline to optimize and compile the model for deployment on the **B1600 DPU**.

---

## 📂 Repository Layout

```text
vitis-ai-yolov3-tiny/
├── arch.json
├── train_yolov3.py
├── eval_yolov3tiny.py
├── compile_B1600_tiny-yolov3.sh
│
├── cfg/
├── common/
├── configs/
├── data/
├── float/
├── logs/
├── tools/
├── yolo3/
└── (other support files)
```

---

## 🛠️ Getting Started

### 1. Enter Docker

```bash
cd vitis-ai-yolov3-tiny
```

```bash
./docker_run.sh cpu
# or
./docker_run.sh gpu
```

---

### 2. Training

```bash
python train_yolov3.py     --anchors_path configs/tiny_yolo3_anchors.txt     --classes_path configs/lane_class.txt     --annotation_file data/lane_detection/train/_annotations.txt     --model_input_shape 256x256     --batch_size 16
```

---

### 3. Quantization & Evaluation

```bash
python eval_yolov3tiny.py     --model_path logs/yolov3_tiny/epXXX-lossXX-val_lossXX.h5     --anchors_path configs/tiny_yolo3_anchors.txt     --classes_path configs/lane_class.txt     --annotation_file data/lane_detection/train/_annotations.txt     --quantize     --eval_quant
```

---

### 4. Compilation

```bash
./compile_B1600_tiny-yolov3.sh
```

---

## 📌 Pipeline Summary

| Step | Script | Output |
|------|--------|---------|
| Training | train_yolov3.py | Floating-point .h5 |
| Quantization | eval_yolov3tiny.py | Quantized model |
| Compilation | compile_B1600_tiny-yolov3.sh | xmodel |
