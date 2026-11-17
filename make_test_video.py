"""
테스트용 공 낙하 영상 생성기

실제 영상을 찍기 전에 분석 툴이 제대로 동작하는지 확인하려고 만든 스크립트입니다.
물리 법칙(등가속도 운동)대로 움직이는 주황색 공과, 길이를 아는 기준 자를 그려서 mp4로 저장합니다.

사용 예:
    python make_test_video.py                      # 자유낙하 (g = 9.80665)
    python make_test_video.py --vx 1.2 --vy 2.0    # 비스듬히 던진 포물선 운동
"""
import argparse
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="테스트용 공 낙하 영상 만들기")
    ap.add_argument("--out", default="test_drop.mp4")
    ap.add_argument("--fps", type=float, default=120.0, help="영상 프레임레이트")
    ap.add_argument("--g", type=float, default=9.80665, help="시뮬레이션에 쓸 중력가속도 (m/s²)")
    ap.add_argument("--vx", type=float, default=0.0, help="수평 초기속도 (m/s, 오른쪽 +)")
    ap.add_argument("--vy", type=float, default=0.0, help="수직 초기속도 (m/s, 위쪽 +)")
    ap.add_argument("--ppm", type=float, default=300.0, help="1 m가 몇 픽셀인지")
    ap.add_argument("--noise", type=float, default=6.0, help="영상 노이즈 세기")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    W, H = 640, 520
    floor_px = 480                     # 바닥 위치 (픽셀)
    x0_px, y0_px = 140 if a.vx else 320, 60   # 공 시작 위치
    r_px = 11
    hold = 0.25                        # 손에서 놓기 전 정지 시간 (s)
    total = 1.2                        # 영상 길이 (s)

    # 기준 자: 화면 왼쪽에 1.000 m 길이
    ruler_x, ruler_top = 40, 70
    ruler_bot = ruler_top + int(round(1.0 * a.ppm))

    # 배경 (약한 그라데이션)
    yy, xx = np.mgrid[0:H, 0:W]
    bg = np.dstack([180 + 20 * xx / W, 185 + 15 * yy / H, 190 + 10 * xx / W]).astype(np.uint8)

    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    n = int(total * a.fps)
    x, y = x0_px / a.ppm, -y0_px / a.ppm      # 미터 단위, 위쪽 +
    vx, vy = 0.0, 0.0
    dt = 1.0 / a.fps
    released = False
    for i in range(n):
        t = i * dt
        if t >= hold and not released:
            released, vx, vy = True, a.vx, a.vy
        if released:
            # 해석해를 작은 단계로 적분 (바닥에서 튕김 포함)
            sub = 20
            for _ in range(sub):
                h = dt / sub
                x += vx * h
                y += vy * h - 0.5 * a.g * h * h
                vy -= a.g * h
                if -y * a.ppm + r_px >= floor_px and vy < 0:
                    y = -(floor_px - r_px) / a.ppm
                    vy = -vy * 0.55
                    vx *= 0.8
        img = bg.copy()
        cv2.rectangle(img, (0, floor_px), (W, H), (120, 120, 120), -1)
        # 자
        cv2.rectangle(img, (ruler_x - 6, ruler_top), (ruler_x + 6, ruler_bot), (235, 235, 235), -1)
        for k in range(11):
            yk = ruler_top + int(round(k * 0.1 * a.ppm))
            cv2.line(img, (ruler_x - 6, yk), (ruler_x + 10, yk), (40, 40, 40), 2)
        cv2.putText(img, "1.00 m", (ruler_x + 14, (ruler_top + ruler_bot) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        # 공 (주황색, BGR)
        cx, cy = x * a.ppm, -y * a.ppm
        # shift=4: 1/16 픽셀 단위로 그려서 정수 반올림 오차를 줄임
        cv2.circle(img, (int(round(cx * 16)), int(round(cy * 16))), r_px * 16, (0, 140, 255), -1, lineType=cv2.LINE_AA, shift=4)
        noisy = img.astype(np.float32) + rng.normal(0, a.noise, img.shape)
        vw.write(np.clip(noisy, 0, 255).astype(np.uint8))
    vw.release()
    print(f"저장: {a.out}  ({n} 프레임, {a.fps:g} fps)")
    print(f"기준 자 양 끝 픽셀 좌표: {ruler_x},{ruler_top} ~ {ruler_x},{ruler_bot}  (길이 1.0 m)")
    print(f"분석 예: python ball_tracker.py {a.out} --color orange --scale-points {ruler_x},{ruler_top},{ruler_x},{ruler_bot} --scale-length 1.0")


if __name__ == "__main__":
    main()
