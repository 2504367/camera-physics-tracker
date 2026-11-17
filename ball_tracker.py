"""
스마트폰 카메라 기반 물리 실험 데이터 자동 분석 툴 (1일차: 공 추적)

영상 속 공을 색깔로 찾아서 프레임별 픽셀 좌표를 CSV로 저장합니다.

처리 순서
    1. 프레임을 HSV 색공간으로 바꾸고, 공 색 범위만 남기는 마스크를 만듦
    2. 열림·닫힘 연산으로 잡티를 지우고 윤곽선을 찾음
    3. 윤곽선 중 면적이 충분하고 원에 가까운 것을 공으로 선택
       (후보가 여러 개면 직전 프레임 위치와 가까운 것을 우선)
    4. 무게중심(모멘트)으로 공의 중심 좌표를 구해 CSV로 저장

사용 예
    python ball_tracker.py test_drop.mp4 --color orange
    python ball_tracker.py my_drop.mp4 --hsv 5,120,120,22,255,255 --save-mask
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

# OpenCV HSV: H 0~179, S 0~255, V 0~255
COLOR_PRESETS = {
    "orange": [((5, 120, 120), (22, 255, 255))],
    "yellow": [((22, 100, 120), (35, 255, 255))],
    "green":  [((40, 80, 60), (85, 255, 255))],
    "blue":   [((95, 120, 60), (130, 255, 255))],
    "red":    [((0, 120, 90), (6, 255, 255)), ((170, 120, 90), (179, 255, 255))],
    "pink":   [((140, 80, 120), (170, 255, 255))],
}


def find_ball(frame, ranges, prev, min_area, kernel):
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1e18
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (per * per) if per > 0 else 0
        if circ < 0.45:          # 공은 원에 가까움 (움직임 번짐 고려해 느슨하게)
            continue
        M = cv2.moments(c)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        score = area * circ
        if prev is not None:     # 직전 위치와 가까울수록 우선
            score -= 4.0 * np.hypot(cx - prev[0], cy - prev[1]) ** 2
        if score > best_score:
            (_, _), r = cv2.minEnclosingCircle(c)
            best, best_score = (cx, cy, r), score
    return best, mask


def main():
    ap = argparse.ArgumentParser(description="영상 속 공을 추적해 프레임별 픽셀 좌표를 저장합니다.")
    ap.add_argument("video")
    ap.add_argument("--color", choices=COLOR_PRESETS.keys(), help="공 색상 프리셋")
    ap.add_argument("--hsv", help="직접 HSV 범위: h1,s1,v1,h2,s2,v2")
    ap.add_argument("--fps", type=float, help="실제 촬영 fps (기본: 영상 파일 정보)")
    ap.add_argument("--min-area", type=float, default=30, help="공으로 인정할 최소 면적 (px²)")
    ap.add_argument("--save-mask", action="store_true", help="확인용으로 중간 프레임의 마스크 이미지 저장")
    ap.add_argument("--out", help="결과 폴더 (기본: 영상이름_result)")
    args = ap.parse_args()

    if args.hsv:
        v = [float(t) for t in args.hsv.split(",")]
        ranges = [(v[0:3], v[3:6])]
    elif args.color:
        ranges = COLOR_PRESETS[args.color]
    else:
        sys.exit("공 색상을 정해 주세요: --color orange 또는 --hsv h1,s1,v1,h2,s2,v2")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"영상을 열 수 없어요: {args.video}")
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    out = args.out or os.path.splitext(os.path.basename(args.video))[0] + "_result"
    os.makedirs(out, exist_ok=True)

    rows, prev, i = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hit, mask = find_ball(frame, ranges, prev, args.min_area, kernel)
        if hit:
            prev = hit[:2]
            rows.append([i, i / fps, hit[0], hit[1], hit[2]])
        else:
            rows.append([i, i / fps, None, None, None])
        if args.save_mask and i == n_total // 2:
            cv2.imwrite(os.path.join(out, "mask_sample.png"), mask)
            cv2.imwrite(os.path.join(out, "frame_sample.png"), frame)
        i += 1
    cap.release()

    path = os.path.join(out, "track_px.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "x_px", "y_px", "radius_px"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.5f}"] + ["" if v is None else f"{v:.2f}" for v in r[2:]])

    detected = sum(1 for r in rows if r[2] is not None)
    print(f"프레임 {len(rows)}개 중 {detected}개에서 공 검출 (fps {fps:g})")
    print(f"저장: {path}")
    if detected == 0:
        print("공을 찾지 못했어요. --save-mask로 마스크를 확인하고 색 범위를 조정해 보세요.")


if __name__ == "__main__":
    main()
