"""
스마트폰 카메라 기반 물리 실험 데이터 자동 분석 툴 (2일차: 보정과 g 계산)

영상 속 공을 색깔로 추적해서 프레임별 위치를 뽑고,
픽셀 좌표를 실제 길이(m)로 바꾼 뒤 속도·가속도·중력가속도 g를 계산합니다.

처리 순서
    1. 색상 추적: HSV 색 범위로 공을 찾고(마스킹), 윤곽선 중 가장 공다운 것을 고름
    2. 보정(캘리브레이션): 화면 속 길이를 아는 물체(자)의 양 끝 픽셀 → m/px 비율
    3. 분석: y(t) = ½·a·t² + v0·t + y0 2차 곡선 맞춤 → g = −a (표준오차 포함)
             국소 2차 회귀(Savitzky–Golay 방식)로 노이즈를 줄이며 프레임별 속도·가속도 계산
    4. 결과: CSV, 그래프(PNG), 요약(JSON)

사용 예
    # 공이 떨어지는 구간(초)을 --start, --end로 지정
    python ball_tracker.py test_drop.mp4 --color orange --scale-points 40,70,40,370 --scale-length 1.0 --start 0.26 --end 0.75
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

G_THEORY = 9.80665

# OpenCV HSV: H 0~179, S 0~255, V 0~255
COLOR_PRESETS = {
    "orange": [((5, 120, 120), (22, 255, 255))],
    "yellow": [((22, 100, 120), (35, 255, 255))],
    "green":  [((40, 80, 60), (85, 255, 255))],
    "blue":   [((95, 120, 60), (130, 255, 255))],
    "red":    [((0, 120, 90), (6, 255, 255)), ((170, 120, 90), (179, 255, 255))],
    "pink":   [((140, 80, 120), (170, 255, 255))],
}


# ----------------------------------------------------------------------------
# 1. 설정 (색상 범위, 픽셀-미터 비율)
# ----------------------------------------------------------------------------
def build_config(args):
    cfg = {}
    if args.color:
        cfg["hsv_ranges"] = [[list(lo), list(hi)] for lo, hi in COLOR_PRESETS[args.color]]
    if args.hsv:
        v = [float(t) for t in args.hsv.split(",")]
        cfg["hsv_ranges"] = [[v[0:3], v[3:6]]]
    if "hsv_ranges" not in cfg:
        sys.exit("공 색상을 정해 주세요: --color orange 또는 --hsv h1,s1,v1,h2,s2,v2")
    if args.px_per_m:
        cfg["px_per_m"] = args.px_per_m
    elif args.scale_points and args.scale_length:
        x1, y1, x2, y2 = [float(t) for t in args.scale_points.split(",")]
        cfg["px_per_m"] = float(np.hypot(x2 - x1, y2 - y1)) / args.scale_length
    else:
        sys.exit("길이 기준을 정해 주세요: --scale-points x1,y1,x2,y2 + --scale-length 길이(m), 또는 --px-per-m")
    return cfg


# ----------------------------------------------------------------------------
# 2. 프레임별 공 찾기
# ----------------------------------------------------------------------------
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


def track(path, cfg, args):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"영상을 열 수 없어요: {path}")
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps or file_fps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rows, prev, i = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        if (args.start is None or t >= args.start) and (args.end is None or t <= args.end):
            hit, _ = find_ball(frame, cfg["hsv_ranges"], prev, args.min_area, kernel)
            if hit:
                prev = hit[:2]
                rows.append((i, t, hit[0], hit[1], hit[2]))
            else:
                rows.append((i, t, np.nan, np.nan, np.nan))
        i += 1
    cap.release()
    if not rows:
        sys.exit("분석할 프레임이 없어요. --start/--end 값을 확인해 주세요.")
    return np.array(rows, dtype=float), fps, file_fps


# ----------------------------------------------------------------------------
# 3. 물리량 계산
# ----------------------------------------------------------------------------
def local_quadratic(t, y, w):
    """각 점 주변 w개 점에 2차식을 맞춰 위치·속도·가속도를 구한다 (Savitzky–Golay 방식).
    단순 차분보다 노이즈에 강하고, 양 끝에서도 창을 한쪽으로 밀어서 치우침 없이 계산한다."""
    n, w = len(t), max(3, min(w, len(t)))
    pos, vel, acc = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        lo = min(max(0, i - w // 2), n - w)
        tt, yy = t[lo:lo + w] - t[i], y[lo:lo + w]
        c2, c1, c0 = np.polyfit(tt, yy, 2)
        pos[i], vel[i], acc[i] = c0, c1, 2 * c2
    return pos, vel, acc


def analyze(seg, ppm, smooth):
    t = seg[:, 1] - seg[0, 1]
    x = (seg[:, 2] - seg[0, 2]) / ppm               # 오른쪽 +
    y = -(seg[:, 3] - seg[0, 3]) / ppm              # 위쪽 + (화면 좌표는 아래쪽이 +)

    # 2차 곡선 맞춤 y = c2 t² + c1 t + c0 → 가속도 = 2 c2
    (c2, c1, c0), cov = np.polyfit(t, y, 2, cov=True)
    g_fit = -2 * c2
    g_se = 2 * np.sqrt(cov[0, 0])
    y_hat = np.polyval([c2, c1, c0], t)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    r2 = 1 - ss_res / ss_tot
    bx, _ = np.polyfit(t, x, 1)                     # 수평 방향은 등속 운동 → 1차 맞춤

    # 국소 2차 회귀로 속도(창 smooth개), 가속도(창 2·smooth+1개) 계산
    ys, vy, _ = local_quadratic(t, y, smooth)
    xs, vx, _ = local_quadratic(t, x, smooth)
    _, _, ay = local_quadratic(t, y, 2 * smooth + 1)

    return {
        "t": t, "x": x, "y": y, "y_fit": y_hat, "vx": vx, "vy": vy, "ay": ay,
        "g_fit": float(g_fit), "g_se": float(g_se), "r2": float(r2),
        "v0y": float(c1), "vx_fit": float(bx),
        "rmse_mm": float(np.sqrt(ss_res / len(t)) * 1000),
        "ay_mean": float(np.mean(ay)),
        "fit_coeffs": [float(c2), float(c1), float(c0)],
    }


# ----------------------------------------------------------------------------
# 4. 결과 저장
# ----------------------------------------------------------------------------
def save_csv(path, data, seg_range, ppm, res):
    s, e = seg_range
    used_rows = [k for k in range(s, e) if not np.isnan(data[k, 2])]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_video_s", "x_px", "y_px", "radius_px", "used",
                    "t_s", "x_m", "y_m", "y_fit_m", "vx_mps", "vy_mps", "ay_mps2"])
        for k, row in enumerate(data):
            used = k in used_rows
            base = [int(row[0]), f"{row[1]:.5f}"] + ["" if np.isnan(v) else f"{v:.2f}" for v in row[2:5]] + [int(used)]
            if used:
                j = used_rows.index(k)
                base += [f"{res['t'][j]:.5f}", f"{res['x'][j]:.5f}", f"{res['y'][j]:.5f}", f"{res['y_fit'][j]:.5f}",
                         f"{res['vx'][j]:.4f}", f"{res['vy'][j]:.4f}", f"{res['ay'][j]:.3f}"]
            w.writerow(base)


def save_plot(path, res, title):
    t = res["t"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    ax = axs[0]
    ax.plot(t, res["y"], "o", ms=3, label="measured")
    ax.plot(t, res["y_fit"], "-", lw=2, label=f"fit: g = {res['g_fit']:.3f} m/s²")
    ax.set(xlabel="t (s)", ylabel="y (m, up +)", title="Position")
    ax.legend(); ax.grid(alpha=.3)

    ax = axs[1]
    ax.plot(t, res["vy"], "o", ms=3, label="vy (local fit)")
    ax.plot(t, res["v0y"] - res["g_fit"] * t, "-", lw=2, label="vy from fit")
    ax.set(xlabel="t (s)", ylabel="v (m/s)", title="Velocity")
    ax.legend(); ax.grid(alpha=.3)

    ax = axs[2]
    ax.plot(t, res["ay"], "o", ms=3, label="ay (local fit)")
    ax.axhline(-res["g_fit"], color="C1", lw=2, label=f"fit  {-res['g_fit']:.3f}")
    ax.axhline(-G_THEORY, color="k", ls="--", lw=1, label=f"theory {-G_THEORY:.3f}")
    ax.set(xlabel="t (s)", ylabel="a (m/s²)", title="Acceleration")
    ax.set_ylim(-G_THEORY * 2.2, G_THEORY * 0.6)
    ax.legend(); ax.grid(alpha=.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="영상 속 공을 추적해 위치·속도·가속도·g를 계산합니다.")
    ap.add_argument("video")
    ap.add_argument("--color", choices=COLOR_PRESETS.keys(), help="공 색상 프리셋")
    ap.add_argument("--hsv", help="직접 HSV 범위: h1,s1,v1,h2,s2,v2")
    ap.add_argument("--scale-points", help="기준 물체 양 끝 픽셀: x1,y1,x2,y2")
    ap.add_argument("--scale-length", type=float, help="기준 물체 실제 길이 (m)")
    ap.add_argument("--px-per-m", type=float, help="1 m당 픽셀 수 (직접 지정)")
    ap.add_argument("--fps", type=float, help="실제 촬영 fps (슬로모션이면 꼭 지정)")
    ap.add_argument("--start", type=float, help="분석 시작 시각 (s, 영상 기준) - 공을 놓은 직후")
    ap.add_argument("--end", type=float, help="분석 끝 시각 (s, 영상 기준) - 바닥에 닿기 직전")
    ap.add_argument("--smooth", type=int, default=7, help="속도 계산에 쓰는 국소 2차 회귀 창 크기 (가속도는 2배+1)")
    ap.add_argument("--min-area", type=float, default=30, help="공으로 인정할 최소 면적 (px²)")
    ap.add_argument("--out", help="결과 폴더 (기본: 영상이름_result)")
    args = ap.parse_args()

    cfg = build_config(args)
    ppm = cfg["px_per_m"]
    data, fps, file_fps = track(args.video, cfg, args)
    detected = int(np.sum(~np.isnan(data[:, 2])))
    print(f"프레임 {len(data)}개 중 {detected}개에서 공 검출  (분석 fps {fps:g}, {ppm:.1f} px/m)")

    seg_range = (0, len(data))
    data_seg = data[~np.isnan(data[:, 2])]
    if len(data_seg) < 5:
        sys.exit("분석에 쓸 점이 5개보다 적어요.")

    res = analyze(data_seg, ppm, max(1, args.smooth))
    err = abs(res["g_fit"] - G_THEORY) / G_THEORY * 100
    tc = 1.96 if len(data_seg) > 30 else 2.1
    ci = (res["g_fit"] - tc * res["g_se"], res["g_fit"] + tc * res["g_se"])

    out = args.out or os.path.splitext(os.path.basename(args.video))[0] + "_result"
    os.makedirs(out, exist_ok=True)
    save_csv(os.path.join(out, "track.csv"), data, seg_range, ppm, res)
    save_plot(os.path.join(out, "analysis.png"), res, os.path.basename(args.video))
    summary = {
        "video": args.video, "fps_used": fps, "px_per_m": ppm,
        "frames_total": len(data), "frames_detected": detected, "points_used": len(data_seg),
        "duration_s": float(res["t"][-1]),
        "g_fit": res["g_fit"], "g_standard_error": res["g_se"], "g_ci95": list(ci),
        "g_theory": G_THEORY, "relative_error_pct": err,
        "r2": res["r2"], "rmse_mm": res["rmse_mm"],
        "initial_vy": res["v0y"], "horizontal_v": res["vx_fit"],
        "mean_ay_local_fit": res["ay_mean"],
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("========== 분석 결과 ==========")
    print(f"사용한 점      : {len(data_seg)}개, {res['t'][-1]:.3f} s")
    print(f"중력가속도 g   : {res['g_fit']:.3f} ± {res['g_se']:.3f} m/s²  (95% 구간 {ci[0]:.3f} ~ {ci[1]:.3f})")
    print(f"이론값 대비    : {G_THEORY:.3f} m/s², 오차율 {err:.2f} %"
          + ("  → 이론값이 95% 구간 안" if ci[0] <= G_THEORY <= ci[1] else "  → 이론값이 95% 구간 밖 (계통 오차 의심)"))
    print(f"곡선 맞춤 품질 : R² = {res['r2']:.5f}, 잔차 RMS = {res['rmse_mm']:.2f} mm")
    print(f"초기 속도      : 수직 {res['v0y']:.3f} m/s, 수평 {res['vx_fit']:.3f} m/s")
    print(f"결과 폴더      : {out}/ (track.csv, analysis.png, summary.json)")
    if res["r2"] < 0.99:
        print("\n참고: R²가 낮아요. 공을 놓기 전이나 바닥에 닿은 뒤 프레임이 섞였을 수 있어요. --start, --end로 구간을 좁혀 보세요.")


if __name__ == "__main__":
    main()
