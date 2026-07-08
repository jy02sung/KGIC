# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**성균관대(SKKU) 자율주행 대회 — Ultra96v2 AI칩 자율주행 프로젝트.**

Xilinx **Ultra96-v2 (PYNQ)** 보드의 DPU로 YOLOv3-tiny 차선검출을 돌려 소형 차량을 자율주행시키는 프로젝트. 카메라 프레임을 BEV(Bird's-Eye-View)로 전처리해 DPU에서 추론하고, 조향/구동 모터를 폐루프로 제어한다.

> 이 폴더는 물리적으로 `HEVEN/` 레포 안에 있지만, HEVEN 동아리의 **화성 대회**(실제 차량 제작 자작자율차 부문)나 **영광 대회**(EV 경주 부문)와는 **무관한 별도 프로젝트**다. HEVEN 동아리 작업은 상위 `HEVEN/CLAUDE.md` 참조.

## 하드웨어 플랫폼

- **보드**: Xilinx Ultra96-v2, PYNQ 이미지, DPU(Deep Processing Unit)로 양자화 모델 추론
- **모델**: `tiny-yolov3_256.xmodel` (DPU용 컴파일 모델). 양자화 소스 `quantized_yolov3_tiny.h5`는 상위 HEVEN 폴더에 위치
- **입력 해상도**: 256 (BEV 전처리 후 resize)
- 보드 접속/네트워크/데이터 수집 방법은 **memory 참조** — Ultra96 USB ECM 가젯, SSH/rsync, SKKU wifi, 아이폰 핫스팟 IP 관련 메모리가 이 프로젝트에 이관되어 있음

## 폴더 내용

| 항목 | 설명 |
|------|------|
| `data_collection.ipynb` | 수동주행(키보드 WASD) + 프레임 저장(라벨 speed/steer). **완벽 동작 검증됨 = 하드웨어 계층 ground truth** |
| `collected_data/` | 촬영 원본 데이터셋 (타임스탬프 폴더, 256 원본·BEV 아님). 보드에서 rsync로 수집 |
| `Preprocessed_Dataset/` | BEV 전처리 완료 데이터셋 (타임스탬프 폴더) |
| `train/` | 학습용 라벨링 데이터 (COCO json + `speed:*_steer:*` 파일명 규약, BEV 처리됨) |
| `preprocessing.py` / `preprocessing_recursive.py` | BEV 변환 전처리 (`BEVProcessor`, cv2 getPerspectiveTransform) |
| `*.coco.zip` | Roboflow 등에서 내보낸 COCO 라벨 데이터셋 아카이브 |
| `tiny-yolov3_256.xmodel` | DPU 배포용 컴파일 모델 |

## 핵심 기술 맥락

**두 단계 파이프라인**
- `data_collection.ipynb` — 수동주행 데이터 수집. **하드웨어 계층의 기준(ground truth)**. 프레임은 256 원본 저장, 학습 전 별도 BEV 전처리.
- `SoC_Driving.ipynb` (보드) — 자율주행(YOLOv3-tiny 차선검출). 문제 많았음 → `SoC_Driving_v2.ipynb`가 작동본.

**하드웨어 계층 = data_collection 기준으로 통일** (사용자 지시)
- 모터: 조향 motor_4(좌)/motor_5(우), 구동 motor_2·3(좌)/motor_0·1(우). set_left_speed 전진 = motor_2 enable.
- ADC 조향 포텐셔미터: 우 **1512** / 좌 **2338** (map_value로 ±20, 폐루프). 다른 파일 값(1758/2241, 220/1045)은 신뢰 불가.
- 주의: 캐노니컬 `driving/motor_controller.py`는 모터매핑·좌구동방향이 data_collection과 **반대** → 배선 확정 전엔 캐노니컬 하드웨어 로직 쓰지 말 것.

**비전 / BEV** (모델이 BEV 전처리로 학습됨 → 추론도 BEV 필수)
- 정식 참조: `driving/image_processor.py` (SKKU AutomationLab). 파이프라인: `BEV → ROI(cutting_idx=300) → resize 256 → DPU`.
- BEV 좌표(640×480 기준): src=[[238,316],[402,313],[501,476],[155,476]], dst=[[w·0.3,0],[w·0.7,0],[w·0.7,h],[w·0.3,h]].
- 각도규약 함정: `calculate_angle`는 정면차선 = 90° 반환. bang-bang은 sign만 씀.

**작동 버전**: `SoC_Driving_v2.ipynb` = data_collection 하드웨어 + 캐노니컬 BEV 비전 + vision_to_target(차선중심→±20). 튜닝노브: STEER_DIR(±1), STEER_GAIN, DRIVE_SPEED(15), REF_X(128). 첫 테스트는 바퀴 띄우고.

## Working in This Directory

Claude can help with:
- **데이터 파이프라인**: collected_data 수집(rsync), BEV 전처리, COCO 라벨 관리, 학습셋 구성
- **모델**: YOLOv3-tiny 양자화/컴파일(xmodel), DPU 추론 코드
- **주행 코드**: 보드 `SoC_Driving_v2.ipynb` 튜닝 (모터/ADC/BEV/조향 게인)
- **보드 운영**: Ultra96 접속(USB ECM/SSH), 데이터 수집·백업 (memory의 Ultra96 레퍼런스 참조)
- **Python 환경**: pip 대신 **uv** 사용 (memory 참조)
