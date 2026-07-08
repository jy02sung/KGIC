import os
import cv2
import numpy as np


class BEVProcessor:
    @staticmethod
    def warp_image(image, pts_src, pts_dst):
        """주어진 원본 좌표와 대상 좌표를 기반으로 이미지를 변환합니다."""
        (h, w) = image.shape[:2]
        transform_matrix = cv2.getPerspectiveTransform(np.float32(pts_src), np.float32(pts_dst))
        warped_image = cv2.warpPerspective(image, transform_matrix, (w, h))
        return warped_image

    def bird_convert(self, img, FB):
        """FB 타입에 따라 src_mat 및 dst_mat 설정 후 BEV 변환."""
        (h, w) = img.shape[:2]
        dst_mat = {
            "FRONT": [[round(w * 0.3), 0], [round(w * 0.7), 0], [round(w * 0.7), h], [round(w * 0.3), h]],
            "REAR": [[round(w * 0.2), round(h * 0.15)], [round(w * 0.8), round(h * 0.15)], [round(w * 0.8), h], [round(w * 0.2), h]],
        }

        # 원본 보정값은 640x480 기준 → 실제 데이터(320x240)에 맞춰 0.5배 스케일
        src_mat = [[119, 158], [201, 156], [250, 238], [77, 238]]

        if FB not in dst_mat:
            print(f"Invalid FB type: {FB}")
            return None

        img_warped = self.warp_image(img, src_mat, dst_mat[FB])
        return img_warped


# 입력 및 출력 폴더 설정
input_folder = "./collected_data"                 # 원본 이미지 루트 (재귀 탐색)
output_folder = "./Preprocessed_Dataset"          # 처리된 이미지 저장 루트

# BEV 프로세서 초기화
bev_processor = BEVProcessor()

# FB 타입 정의
FB_TYPE = "FRONT"  # "FRONT" 또는 "REAR" 선택 가능

# 처리 파라미터
resize_dim = (256, 256)  # 리사이즈 크기
cutting_idx = 150        # 자를 시작 인덱스 (640x480 기준 300 → 320x240에 맞춰 0.5배)

# 재귀적으로 모든 .png 파일 처리 (폴더 구조 및 원본 파일명 보존)
processed = 0
failed = 0
for root, dirs, files in os.walk(input_folder):
    # Jupyter 체크포인트 폴더는 건너뜀
    if ".ipynb_checkpoints" in dirs:
        dirs.remove(".ipynb_checkpoints")

    for filename in files:
        if not filename.lower().endswith(".png"):
            continue

        filepath = os.path.join(root, filename)

        image = cv2.imread(filepath)
        if image is None:
            print(f"Cannot read image: {filepath}")
            failed += 1
            continue

        bev_image = bev_processor.bird_convert(image, FB_TYPE)
        if bev_image is None:
            print(f"Failed to apply BEV for {filepath}")
            failed += 1
            continue

        cut_image = bev_image[cutting_idx:, :]
        final_image = cv2.resize(cut_image, resize_dim)

        # 입력 루트 기준 상대 경로를 그대로 출력 루트에 재현
        rel_path = os.path.relpath(filepath, input_folder)
        output_path = os.path.join(output_folder, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cv2.imwrite(output_path, final_image)
        processed += 1

print(f"\nDone. processed={processed}, failed={failed}")
print(f"Output root: {os.path.abspath(output_folder)}")
