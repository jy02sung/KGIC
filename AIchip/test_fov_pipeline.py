"""
와이드(16:9 등)로 찍은 이미지를 현재 주행 파이프라인(SoC_Driving_v2 ImageProcessor 기준:
BEV 원근변환 -> ROI 하단 crop -> 256x256 리사이즈)에 넣기 전에, 두 가지 방식을 같은 사진에
적용해 나란히 비교해보는 도구.

  A) squish  : 16:9 -> 640x480 으로 "가로세로 비율 무시하고" 강제로 리사이즈(비등방 압축) 후,
               기존 캐노니컬 보정값(src_mat, cutting_idx=300)을 그대로 적용.
  B) uniform : 16:9 비율은 유지한 채 높이만 480에 맞춰 축소(등방 축소, 폭은 자동으로 640보다
               커짐 = 더 넓은 실세계 폭을 담음) 후, 보정값을 새 폭/높이 비율에 맞게 재계산해 적용.

두 방식 모두 마지막엔 target_size(기본 256) 정사각형으로 리사이즈해서 실제 DPU 입력과
같은 모양으로 맞춘다. 결과는 이미지별로 저장되며, 어느 쪽이 차선 형태를 덜 왜곡하는지
눈으로 비교할 수 있다.

사용법:
    python test_fov_pipeline.py ./fov_test_images -o ./fov_test_output
"""

import argparse
import os
import glob

import cv2
import numpy as np

# ==================== 캐노니컬 보정값 (640x480 기준, driving/image_processor.py / SoC_Driving_v2 동일) ====================
REF_SIZE = (640, 480)  # (w, h)
REF_SRC = [[238, 316], [402, 313], [501, 476], [155, 476]]  # 좌상, 우상, 우하, 좌하
REF_CUTTING_IDX = 300
REF_SRC_RATIO = [(x / REF_SIZE[0], y / REF_SIZE[1]) for x, y in REF_SRC]
REF_CUTTING_RATIO = REF_CUTTING_IDX / REF_SIZE[1]

TARGET_SIZE = 256


def dst_mat_for(w, h):
    return [[round(w * 0.3), 0], [round(w * 0.7), 0], [round(w * 0.7), h], [round(w * 0.3), h]]


def warp(img, src_mat, dst_mat):
    h, w = img.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32(src_mat), np.float32(dst_mat))
    return cv2.warpPerspective(img, M, (w, h))


def bev_crop_square(img, src_mat, cutting_idx, target_size):
    h, w = img.shape[:2]
    bev = warp(img, src_mat, dst_mat_for(w, h))
    roi = bev[cutting_idx:, :]
    square = cv2.resize(roi, (target_size, target_size))
    return bev, roi, square


def run_squish(img, target_size=TARGET_SIZE):
    """A) 비율 무시하고 640x480으로 강제 리사이즈 후 기존 절대 보정값 그대로 사용."""
    squished = cv2.resize(img, REF_SIZE)
    bev, roi, square = bev_crop_square(squished, REF_SRC, REF_CUTTING_IDX, target_size)
    return squished, bev, roi, square


def run_uniform(img, target_size=TARGET_SIZE):
    """B) 비율 유지한 채 높이 480에 맞춰 축소(폭은 늘어남) 후 보정값을 새 크기 비율로 재계산."""
    h, w = img.shape[:2]
    new_h = REF_SIZE[1]
    new_w = round(w * new_h / h)
    resized = cv2.resize(img, (new_w, new_h))

    src_mat = [[round(rx * new_w), round(ry * new_h)] for rx, ry in REF_SRC_RATIO]
    cutting_idx = round(new_h * REF_CUTTING_RATIO)
    bev, roi, square = bev_crop_square(resized, src_mat, cutting_idx, target_size)
    return resized, bev, roi, square


def label(panel, text):
    panel = panel.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(panel, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return panel


def resize_h(im, height):
    w = max(1, int(im.shape[1] * height / im.shape[0]))
    return cv2.resize(im, (w, height))


def build_row(panels, labels, height=240):
    return cv2.hconcat([label(resize_h(p, height), t) for p, t in zip(panels, labels)])


def roi_aspect(roi):
    h, w = roi.shape[:2]
    return (w / h) if h else 0.0


def process_folder(input_dir, output_dir, target_size):
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png") for p in glob.glob(os.path.join(input_dir, "**", ext), recursive=True)
    )
    if not paths:
        print(f"입력 폴더에 이미지가 없습니다: {input_dir}")
        return

    report = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"읽기 실패: {path}")
            continue
        h, w = img.shape[:2]

        sq_in, sq_bev, sq_roi, sq_square = run_squish(img, target_size)
        un_in, un_bev, un_roi, un_square = run_uniform(img, target_size)

        row_a = build_row([sq_in, sq_bev, sq_roi, sq_square], ["squish input(640x480)", "BEV", "ROI", "square"])
        row_b = build_row([un_in, un_bev, un_roi, un_square], ["uniform input(%dx%d)" % (un_in.shape[1], un_in.shape[0]), "BEV", "ROI", "square"])
        # 가로 폭이 다를 수 있으니 큰 쪽에 맞춰 패딩
        max_w = max(row_a.shape[1], row_b.shape[1])
        row_a = cv2.copyMakeBorder(row_a, 0, 0, 0, max_w - row_a.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
        row_b = cv2.copyMakeBorder(row_b, 0, 0, 0, max_w - row_b.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
        combined = cv2.vconcat([row_a, row_b])

        squares_side_by_side = cv2.hconcat([
            label(sq_square, "A) squish square"),
            label(un_square, "B) uniform square"),
        ])

        name = os.path.splitext(os.path.relpath(path, input_dir))[0].replace(os.sep, "_")
        cv2.imwrite(os.path.join(output_dir, f"{name}_compare.png"), combined)
        cv2.imwrite(os.path.join(output_dir, f"{name}_squares.png"), squares_side_by_side)

        line = (f"{os.path.relpath(path, input_dir)}: 원본 {w}x{h}  |  "
                f"A) squish ROI aspect={roi_aspect(sq_roi):.2f}  "
                f"B) uniform ROI aspect={roi_aspect(un_roi):.2f} (uniform 입력폭={un_in.shape[1]})")
        print(line)
        report.append(line)

    with open(os.path.join(output_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\n완료: {len(report)}장 처리. 결과: {os.path.abspath(output_dir)}")
    print("*_compare.png : 두 방식의 input/BEV/ROI/square 전체 과정 비교")
    print("*_squares.png : 최종 256x256 정사각형 결과만 나란히 비교 (가장 빠르게 눈으로 판단하기 좋음)")
    print("ROI aspect가 1.0에서 멀수록 마지막 square 리사이즈에서 왜곡이 커진다는 뜻입니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", nargs="?", default="./fov_test_images", help="테스트 이미지 폴더 (16:9 와이드 사진 등)")
    parser.add_argument("-o", "--output_dir", default="./fov_test_output", help="비교 이미지 저장 폴더")
    parser.add_argument("--target_size", type=int, default=TARGET_SIZE, help="최종 정사각형 한 변 크기 (기본 256, DPU 입력과 동일해야 함)")
    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir, args.target_size)


if __name__ == "__main__":
    main()
