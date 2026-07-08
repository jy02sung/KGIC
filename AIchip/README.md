# Ultra96-v2 AI칩 자율주행 (SKKU 자율주행 대회)

Xilinx **Ultra96-v2 (PYNQ)** 보드의 DPU에서 YOLOv3-tiny 차선검출 모델을 돌려 소형 차량을 자율주행시키는 프로젝트.
카메라 프레임을 BEV(Bird's-Eye-View)로 전처리해 DPU로 추론하고, 가변저항 피드백 기반 폐루프로 조향을 제어한다.

## 하드웨어

| 항목 | 내용 |
|------|------|
| 보드 | Xilinx Ultra96-v2, PYNQ 이미지, DPU 추론 |
| 모델 | `tiny-yolov3_256.xmodel` (양자화·컴파일된 차선검출 모델, 입력 256×256) |
| 조향 | DC 모터(motor_4=좌 / motor_5=우) + 포텐셔미터 ADC 피드백 (우 1512 / 좌 2338, ±20 스케일) |
| 구동 | motor_2·3(좌) / motor_0·1(우), PWM MMIO 제어 (size=600600 ≈ 2ms @300MHz) |

## 파이프라인

```
카메라(640×480) → BEV 원근변환 → ROI(하단) → 256×256 리사이즈 → DPU(YOLOv3-tiny)
  → 우측 차선 박스 중심 x → 조향각(atan2, 0=직진/음수=좌/양수=우) → 100Hz 폐루프 조향
```

- **비전 루프**: 프레임마다 차선 중심을 검출해 목표 조향각만 갱신
- **제어 스레드(100Hz)**: ADC로 실제 바퀴 위치를 읽어 목표와 비교, bang-bang 방식으로 조향 모터 구동

## 파일 구성

| 파일 | 설명 |
|------|------|
| `SoC_Driving_v2.ipynb` | 기존 메인 주행 노트북 (640×480 캡처, 실측 결과 센서 중앙 크롭이라 화각이 좁음) |
| `SoC_Driving_v3.ipynb` | **화각 확대 버전** (1920×1080 캡처 + 비율유지 BEV + BEV 캘리브레이션 확인 셀). 모터/ADC/폐루프 로직은 v2와 동일, 최초 실행 시 반드시 BEV 캘리브레이션 확인 셀로 `SRC_RATIO` 점검 필요 |
| `SoC_Driving_v2_blackbox.ipynb` | 주행 노트북 + 플라이트 레코더(블랙박스). 매 프레임의 인식/판단/제어 상태를 보드 디스크에 기록 |
| `replay_viewer.html` | 블랙박스 로그 오프라인 분석 뷰어. `run_*` 폴더를 열면 프레임 영상 + 동기화 차트 + 자동 진단 배지 표시 |
| `data_collection.ipynb` | 수동주행(WASD) 데이터 수집. 하드웨어 계층의 ground truth |
| `preprocessing.py` / `preprocessing_recursive.py` | 수집 데이터 BEV 전처리 |
| `tiny-yolov3_256.xmodel` | DPU 배포용 컴파일 모델 |
| `archive/` | 구버전 주행 노트북 보관 |

`collected_data/`(수집 원본), `Preprocessed_Dataset/`(BEV 전처리본), `logs/`(블랙박스 기록)는 용량 문제로 git에서 제외됨 (`.gitignore` 참조).

## 사용법

### 주행

1. 보드에 노트북 업로드 후 Jupyter에서 셀 순서대로 실행
2. **첫 테스트는 반드시 바퀴를 지면에서 띄운 상태로** — 조향 방향 확인 셀, ADC 캘리브레이션 셀로 사전 점검
3. 튜닝 노브: `DRIVE_SPEED`, `STEER_TRIM`(중립 트림), `STEER_DIR`(조향 부호), `REF_X`(차선 기준 x)

### 블랙박스 디버깅

주행 문제가 **인식(YOLO) / 판단(각도 변환) / 제어(폐루프)** 중 어느 레이어인지 사후 분석하는 워크플로우:

```bash
# 1. 보드에서 SoC_Driving_v2_blackbox.ipynb 로 주행 (자동 기록됨)
# 2. 로그 회수
scp -r xilinx@<보드IP>:/home/xilinx/jupyter_notebooks/KGIC/driveCode/logs/run_* ./logs/
# 3. replay_viewer.html 을 브라우저로 열고 run_* 폴더 선택
```

뷰어에서 프레임 단위로 center(인식) / target(판단) / mapped(제어) 그래프가 동기화되어 표시되며,
미검출·center 점프·오차 미수렴 같은 이상 징후를 자동 배지로 알려준다.

## 조향각 규약

- 조향 스케일: **−20(풀좌) ~ +20(풀우)**, 각도(°)가 아닌 포텐셔미터 실측 매핑값
- 비전 각도: `atan2` 기반, **0 = 직진, 음수 = 좌, 양수 = 우** (수직선 기준, 전 구간 연속)
