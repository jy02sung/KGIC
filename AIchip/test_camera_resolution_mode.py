"""
카메라에 cv2.VideoCapture로 cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT, 640, 480)을 걸면
실제로 CROP(양옆 잘라냄)인지 SQUISH(비율 무시하고 짜부시켜 리사이즈)인지,
아니면 그냥 별도의 640x480 네이티브 모드로 전환되는 건지 확인하는 보드용 진단 스크립트.

SoC_Driving_v2.ipynb와 똑같이 640x480을 cap.set()으로 요청해 실제로 뭐가 돌아오는지 저장하고,
그 전에 "더 큰 해상도(기본 1920x1080)"를 명시적으로 요청한 프레임을 native 기준으로 삼아
CROP 가설과 SQUISH 가설을 직접 만들어 비교한다.

** cap.set()을 아예 안 부르면 드라이버가 640x480 같은 낮은 기본 모드를 그냥 내려주는 경우가
많아서, 그걸 native로 쓰면 애초에 해상도 변화가 없어 CROP/SQUISH 판정이 무의미해진다.
그래서 native 단계에서도 반드시 --native_width/--native_height 를 명시적으로 요청한다. **

사용법 (보드에서):
    python test_camera_resolution_mode.py
    python test_camera_resolution_mode.py --native_width 1280 --native_height 720  # 1920x1080 미지원 시
    python test_camera_resolution_mode.py --camera_index 0 --warmup 10

주의:
    - native 프레임과 640x480 프레임을 찍는 사이에 카메라/장면이 움직이면 비교가 부정확해집니다.
      스크립트 실행 중에는 카메라와 촬영 대상을 고정해두세요.
    - --warmup 은 해상도 전환 직후 오토 노출/화이트밸런스가 안정될 때까지 버리는 프레임 수입니다.
    - native 단계에서 요청한 해상도가 카메라가 실제로 지원하지 않으면, cap.get()으로 확인한
      실제 반환 해상도가 요청과 다르게 찍히고 스크립트가 경고를 출력합니다. 그 경우 흔한 다른
      해상도(1280x720, 1024x768 등)로 --native_width/--native_height 를 바꿔 다시 시도하세요.

결과 (./camera_probe_output/):
    native.png              - 명시적으로 큰 해상도를 요청해 받은 프레임
    requested_640x480.png   - cap.set(640,480) 이후 실제로 받은 프레임
    synthetic_crop.png      - native를 4:3으로 센터크롭 후 640x480 리사이즈 (가설 A: CROP)
    synthetic_squish.png    - native를 비율 무시하고 640x480으로 강제 리사이즈 (가설 B: SQUISH)
    compare.png             - 넷을 나란히 놓은 비교 이미지 (눈으로 최종 확인용)
    report.txt              - 요청/실제 반환 해상도 + 자동 판정 결과
"""

import argparse
import os

import cv2
import numpy as np


def grab_stable_frame(cap, warmup=5):
    """오토 노출/화이트밸런스가 안정될 때까지 몇 프레임 버리고 마지막 프레임을 반환."""
    frame = None
    for _ in range(max(1, warmup)):
        ret, frame = cap.read()
        if not ret:
            return None
    return frame


def center_crop_to_aspect(img, target_w, target_h):
    h, w = img.shape[:2]
    target_aspect = target_w / target_h
    src_aspect = w / h
    if src_aspect > target_aspect:
        new_w = round(h * target_aspect)
        x0 = (w - new_w) // 2
        cropped = img[:, x0:x0 + new_w]
    else:
        new_h = round(w / target_aspect)
        y0 = (h - new_h) // 2
        cropped = img[y0:y0 + new_h, :]
    return cv2.resize(cropped, (target_w, target_h))


def diff_score(a, b):
    """같은 크기 두 이미지의 평균 밝기 절대차 (작을수록 유사)."""
    a_g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b_g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(a_g - b_g)))


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def resize_h(img, height):
    w = max(1, int(img.shape[1] * height / img.shape[0]))
    return cv2.resize(img, (w, height))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--native_width", type=int, default=1920, help="native 기준으로 명시 요청할 해상도 폭")
    parser.add_argument("--native_height", type=int, default=1080, help="native 기준으로 명시 요청할 해상도 높이")
    parser.add_argument("--warmup", type=int, default=5, help="해상도 전환 후 버릴 프레임 수 (오토 노출 안정화)")
    parser.add_argument("-o", "--output_dir", default="./camera_probe_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 1) native 기준: 명시적으로 큰 해상도를 요청 (cap.set() 없이 그냥 읽으면 드라이버가
    #    640x480 같은 낮은 기본 모드를 내려줄 수 있어 native 기준으로 쓸 수 없음)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.native_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.native_height)
    native = grab_stable_frame(cap, args.warmup)
    if native is None:
        print("네이티브 프레임을 읽지 못했습니다.")
        cap.release()
        return
    nh, nw = native.shape[:2]
    print(f"[native] {args.native_width}x{args.native_height} 요청 -> 실제 반환: {nw}x{nh}")
    if (nw, nh) != (args.native_width, args.native_height):
        print(f"  경고: 요청한 해상도가 그대로 반영되지 않았습니다. 카메라가 이 해상도를 "
              f"지원하지 않을 수 있습니다. --native_width/--native_height 를 카메라가 지원하는 "
              f"다른 값(예: 1280x720)으로 바꿔 다시 시도해보세요.")
    if (nw, nh) == (640, 480):
        print("  경고: native가 640x480으로 나왔습니다 -- 요청한 큰 해상도를 카메라가 지원하지 "
              "않아 640x480으로 폴백했을 가능성이 높습니다. 이 상태로는 CROP/SQUISH 비교가 "
              "무의미합니다 (비교 대상 두 프레임이 사실상 같은 해상도이기 때문).")

    # 2) SoC_Driving_v2와 동일하게 640x480 요청
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    requested = grab_stable_frame(cap, args.warmup)
    cap.release()

    if requested is None:
        print("640x480 요청 후 프레임을 읽지 못했습니다.")
        return
    rh, rw = requested.shape[:2]
    print(f"[requested] cap.set(640,480) 이후 실제 반환 해상도: {rw}x{rh}")

    # 3) native로부터 두 가설(CROP / SQUISH)을 직접 만들어 비교
    synthetic_crop = center_crop_to_aspect(native, 640, 480)
    synthetic_squish = cv2.resize(native, (640, 480))
    requested_cmp = cv2.resize(requested, (640, 480)) if (rw, rh) != (640, 480) else requested

    crop_score = diff_score(requested_cmp, synthetic_crop)
    squish_score = diff_score(requested_cmp, synthetic_squish)

    if abs(crop_score - squish_score) < 2.0:
        verdict = "판정 어려움 (두 가설의 차이가 작음) -- compare.png를 눈으로 확인하세요"
    elif crop_score < squish_score:
        verdict = "CROP (양옆을 잘라내는 쪽에 더 가까움)"
    else:
        verdict = "SQUISH (비율 무시하고 짜부시키는 쪽에 더 가까움)"

    cv2.imwrite(os.path.join(args.output_dir, "native.png"), native)
    cv2.imwrite(os.path.join(args.output_dir, "requested_640x480.png"), requested)
    cv2.imwrite(os.path.join(args.output_dir, "synthetic_crop.png"), synthetic_crop)
    cv2.imwrite(os.path.join(args.output_dir, "synthetic_squish.png"), synthetic_squish)

    panels = [
        label(resize_h(native, 240), f"native {nw}x{nh}"),
        label(resize_h(requested, 240), f"requested 640x480 (actual {rw}x{rh})"),
        label(resize_h(synthetic_crop, 240), f"hypothesis CROP (diff={crop_score:.2f})"),
        label(resize_h(synthetic_squish, 240), f"hypothesis SQUISH (diff={squish_score:.2f})"),
    ]
    cv2.imwrite(os.path.join(args.output_dir, "compare.png"), cv2.hconcat(panels))

    report = (
        f"native: requested {args.native_width}x{args.native_height} -> actual returned {nw}x{nh}\n"
        f"640x480 request -> actual returned  : {rw}x{rh}\n"
        f"diff vs synthetic CROP              : {crop_score:.3f}\n"
        f"diff vs synthetic SQUISH            : {squish_score:.3f}\n"
        f"=> 판정: {verdict}\n"
    )
    with open(os.path.join(args.output_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)
    print(f"결과 저장 위치: {os.path.abspath(args.output_dir)}")
    print("compare.png 를 열어 requested_640x480 이 CROP/SQUISH 가설 중 어느 쪽과 더 비슷한지 눈으로도 확인하세요.")


if __name__ == "__main__":
    main()
