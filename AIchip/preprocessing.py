import os
import cv2
import numpy as np

class BEVProcessor:
    @staticmethod
    def warp_image(image, pts_src, pts_dst):
        """
        주어진 원본 좌표와 대상 좌표를 기반으로 이미지를 변환합니다.
        """
        (h, w) = image.shape[:2]
        transform_matrix = cv2.getPerspectiveTransform(np.float32(pts_src), np.float32(pts_dst))
        warped_image = cv2.warpPerspective(image, transform_matrix, (w, h))
        return warped_image

    def bird_convert(self, img, FB):
        """
        FB 타입에 따라 src_mat 및 dst_mat 설정 후 BEV 변환.
        """
        (h, w) = img.shape[:2]
        dst_mat = {
            "FRONT": [[round(w * 0.3), 0], [round(w * 0.7), 0], [round(w * 0.7), h], [round(w * 0.3), h]],
            "REAR": [[round(w * 0.2), round(h * 0.15)], [round(w * 0.8), round(h * 0.15)], [round(w * 0.8), h], [round(w * 0.2), h]]
        }

        # src_mat는 FB에 따라 설정되거나 외부에서 제공 가능
        src_mat = [[238, 316], [402, 313], [501, 476], [155, 476]]

        if FB not in dst_mat:
            print(f"Invalid FB type: {FB}")
            return None

        img_warped = self.warp_image(img, src_mat, dst_mat[FB])
        return img_warped


# 입력 및 출력 폴더 설정
input_folder = "./"  # 원본 이미지 폴더 경로
output_folder = "./Preprocessed_Test_Dataset/"  # 처리된 이미지 저장 폴더 경로

# 출력 폴더 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# BEV 프로세서 초기화
bev_processor = BEVProcessor()

# FB 타입 정의
FB_TYPE = "FRONT"  # "FRONT" 또는 "REAR" 선택 가능

# 처리 파라미터
resize_dim = (256, 256)  # 리사이즈 크기
cutting_idx = 300  # 자를 시작 인덱스

# 입력 폴더의 모든 .jpg 파일 처리
file_count = 1  # 저장 파일 이름 생성용 카운터
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):  # .jpg 파일만 처리
        filepath = os.path.join(input_folder, filename)

        # 이미지 읽기
        image = cv2.imread(filepath)
        if image is None:
            print(f"Cannot read image: {filepath}")
            continue

        # BEV 적용
        bev_image = bev_processor.bird_convert(image, FB_TYPE)
        if bev_image is None:
            print(f"Failed to apply BEV for {filepath}")
            continue

        # idx 처리: 특정 인덱스 이후로 이미지 자르기
        cut_image = bev_image[cutting_idx:, :]

        # 리사이즈 적용
        final_image = cv2.resize(cut_image, resize_dim)

        # 결과 저장
        output_filename = f"dataset{file_count}.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, final_image)

        print(f"Processed and saved: {output_path}")
        file_count += 1
