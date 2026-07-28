#!/usr/bin/env python3
"""ADC 매핑을 바꾼 뒤 TRACK_CORNERS의 dpsi 값을 다시 재는 도구.

왜 필요한가:
    psi = ∫ PSI_K * mapped dt 이고, mapped는 조향 포텐셔미터 ADC를
    resistance_most_left / resistance_most_right 로 선형 사상한 값이다.
    따라서 그 두 값을 바꾸면 psi 스케일이 통째로 바뀌고, TRACK_CORNERS의
    dpsi 테이블이 실제와 어긋난다.

    이게 위험한 이유는 조용히 틀리기 때문이다. 예외도 경고도 없다.
    코너 진행도 prog = (psi - psi_진입) / dpsi_테이블 가 1에 못 미치고 끝나면,
    정점(d_apex)과 탈출(d_out) 프로파일이 영영 적용되지 않는다.
    실제로 매핑 교정 전 테이블은 좌코너에서 실제의 59~63%였고,
    prog가 0.60에서 멈춰 d_out은 한 번도 쓰이지 않았다.

사용법:
    python3 driveCode/tools/recalc_dpsi.py driveCode/logs/run_YYYYMMDD_HHMMSS

    매핑을 바꾼 뒤 라인을 끄고(RACING_LINE_ENABLE = False) 한 바퀴 돈 로그를 쓸 것.
    라인을 켠 채로 재면 라인이 바꾼 주행이 테이블에 섞여 들어간다.

출력한 dpsi 열을 TRACK_CORNERS에 그대로 붙여넣으면 된다.
"""
import json
import os
import sys

NAMES = ("T1", "T2", "T3", "T4", "T5")


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "telemetry.jsonl")
    with open(path) as f:
        return [json.loads(l) for l in f]


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    rows = load(argv[1])
    if not rows or "seg" not in rows[0]:
        print("seg/psi 필드가 없는 로그입니다. v13 이후 로그를 쓰세요.")
        return 1

    first, last, prog_end = {}, {}, {}
    for r in rows:
        s = r.get("seg")
        if s in NAMES and r.get("prog") is not None:
            first.setdefault(s, r["psi"])
            last[s] = r["psi"]
            prog_end[s] = r["prog"]

    missing = [n for n in NAMES if n not in first]
    if missing:
        print("⚠️  검출되지 않은 코너: %s" % ", ".join(missing))
        print("   코너 판정(CORNER_ENTER)이 매핑 변경으로 안 맞을 수 있습니다.")
        print("   |mapped|의 직선/코너 분포를 먼저 확인하세요.\n")

    print("코너   새 dpsi     이번 로그의 prog 도달   판정")
    for n in NAMES:
        if n not in first:
            print("  %-3s      —              —            미검출" % n)
            continue
        d = last[n] - first[n]
        p = prog_end[n]
        # prog가 1.0 근처면 기존 테이블이 이미 맞다는 뜻.
        verdict = "테이블 정상" if 0.9 <= p <= 1.15 else "테이블 어긋남 → 갱신 필요"
        print("  %-3s  %+8.1f°        %.2f          %s" % (n, d, p, verdict))

    print("\nTRACK_CORNERS에 붙여넣을 dpsi 열:")
    for n in NAMES:
        if n in first:
            print('    ("%s", %+8.1f, ...),' % (n, last[n] - first[n]))

    print("\n주의: 이 값들은 위 로그의 ADC 매핑에서만 유효합니다.")
    print("매핑을 또 바꾸면 다시 재세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
