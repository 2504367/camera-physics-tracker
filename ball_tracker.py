"""
스마트폰 카메라 기반 물리 실험 데이터 자동 분석 툴

영상 속 공을 색깔로 추적해서 프레임별 위치를 뽑고,
픽셀 좌표를 실제 길이(m)로 바꾼 뒤 속도·가속도·중력가속도 g를 계산합니다.

처리 순서
    1. 색상 추적: HSV 색 범위로 공을 찾고(마스킹), 윤곽선 중 가장 공다운 것을 고름
    2. 보정(캘리브레이션): 화면 속 길이를 아는 물체(자)의 양 끝 픽셀 → m/px 비율
    3. 구간 자동 선택: 손에서 놓기 전 정지 구간과 바닥에서 튕긴 뒤 구간을 제외
    4. 분석: y(t) = ½·a·t² + v0·t + y0 2차 곡선 맞춤 → g = −a (표준오차 포함)
             국소 2차 회귀(Savitzky–Golay 방식)로 노이즈를 줄이며 프레임별 속도·가속도 계산
             단위 질량당 역학적 에너지(½v² + g·y)가 보존되는지 확인
    5. 결과: CSV, 그래프(PNG), 요약(JSON), 궤적이 그려진 영상(선택)

사용 예
    # 처음 쓸 때: 화면에서 공과 자 양 끝을 클릭해서 설정 (설정은 calib.json에 저장)
    python ball_tracker.py my_drop.mp4 --pick

    # 설정을 다시 쓰거나 직접 지정
    python ball_tracker.py my_drop.mp4 --calib calib.json
    python ball_tracker.py my_drop.mp4 --color orange --scale-points 40,70,40,370 --scale-length 1.0

    # 슬로모션 영상은 실제 촬영 fps를 꼭 지정 (예: 240fps 촬영)
    python ball_tracker.py slowmo.mp4 --pick --fps 240
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
def hsv_ranges_from_pixel(hsv_patch, dh=10, ds=70, dv=70):
    """클릭한 지점 주변의 평균 색으로 HSV 범위를 만든다. 빨간색처럼 H가 0/179 경계에 걸리면 두 구간으로 나눈다."""
    h, s, v = [float(np.median(hsv_patch[..., k])) for k in range(3)]
    s_lo, s_hi = max(40, s - ds), 255
    v_lo, v_hi = max(40, v - dv), 255
    lo_h, hi_h = h - dh, h + dh
    if lo_h < 0:
        return [((0, s_lo, v_lo), (hi_h, s_hi, v_hi)), ((180 + lo_h, s_lo, v_lo), (179, s_hi, v_hi))]
    if hi_h > 179:
        return [((lo_h, s_lo, v_lo), (179, s_hi, v_hi)), ((0, s_lo, v_lo), (hi_h - 180, s_hi, v_hi))]
    return [((lo_h, s_lo, v_lo), (hi_h, s_hi, v_hi))]


def interactive_pick(frame):
    """첫 프레임을 띄우고 ① 공 ② 자의 한쪽 끝 ③ 자의 다른 쪽 끝을 차례로 클릭받는다."""
    clicks = []
    guide = ["1) 공을 클릭하세요", "2) 기준 자의 한쪽 끝을 클릭하세요", "3) 기준 자의 다른 쪽 끝을 클릭하세요"]
    win = "pick (ESC: cancel)"
    disp = frame.copy()

    def redraw():
        img = disp.copy()
        if len(clicks) < 3:
            cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
            # OpenCV 기본 폰트는 한글을 못 그려서 안내는 영어로 표시
            msg = ["1) click the BALL", "2) click one END of the ruler", "3) click the other END of the ruler"][len(clicks)]
            cv2.putText(img, msg, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        for i, (x, y) in enumerate(clicks):
            cv2.circle(img, (x, y), 6, (0, 0, 255) if i == 0 else (255, 0, 0), 2)
        if len(clicks) >= 3:
            cv2.line(img, clicks[1], clicks[2], (255, 0, 0), 2)
        cv2.imshow(win, img)

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(clicks) < 3:
            clicks.append((x, y))
            if len(clicks) < 3:
                print(guide[len(clicks)])
            redraw()

    print(guide[0])
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while len(clicks) < 3:
        if cv2.waitKey(20) == 27:
            cv2.destroyAllWindows()
            sys.exit("취소했습니다.")
    cv2.waitKey(400)
    cv2.destroyAllWindows()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bx, by = clicks[0]
    patch = hsv[max(0, by - 3):by + 4, max(0, bx - 3):bx + 4]
    ranges = hsv_ranges_from_pixel(patch)
    while True:
        s = input("기준 자의 실제 길이(m)를 입력하세요 (예: 0.3): ").strip()
        try:
            length = float(s)
            if length > 0:
                break
        except ValueError:
            pass
        print("숫자로 입력해 주세요.")
    (x1, y1), (x2, y2) = clicks[1], clicks[2]
    return ranges, [x1, y1, x2, y2], length


def build_config(args, first_frame):
    cfg = {}
    if args.calib and os.path.exists(args.calib) and not args.pick:
        with open(args.calib, encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"설정 불러옴: {args.calib}")
    if args.pick:
        ranges, pts, length = interactive_pick(first_frame)
        cfg = {"hsv_ranges": [[list(map(float, lo)), list(map(float, hi))] for lo, hi in ranges],
               "scale_points": pts, "scale_length": length}
        out = args.calib or "calib.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f"설정 저장: {out}  (다음부터 --calib {out} 로 재사용)")
    if args.color:
        cfg["hsv_ranges"] = [[list(lo), list(hi)] for lo, hi in COLOR_PRESETS[args.color]]
    if args.hsv:
        v = [float(t) for t in args.hsv.split(",")]
        cfg["hsv_ranges"] = [[v[0:3], v[3:6]]]
    if args.scale_points:
        cfg["scale_points"] = [float(t) for t in args.scale_points.split(",")]
    if args.scale_length:
        cfg["scale_length"] = args.scale_length
    if args.px_per_m:
        cfg["px_per_m"] = args.px_per_m

    if "hsv_ranges" not in cfg:
        sys.exit("공 색상을 정해 주세요: --pick, --color orange, 또는 --hsv h1,s1,v1,h2,s2,v2")
    if "px_per_m" not in cfg:
        if "scale_points" in cfg and "scale_length" in cfg:
            x1, y1, x2, y2 = cfg["scale_points"]
            cfg["px_per_m"] = float(np.hypot(x2 - x1, y2 - y1)) / float(cfg["scale_length"])
        else:
            sys.exit("길이 기준을 정해 주세요: --pick, --scale-points + --scale-length, 또는 --px-per-m")
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
    writer, frames_for_video = None, []
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
            if args.save_video:
                frames_for_video.append(frame)
        i += 1
    cap.release()
    if not rows:
        sys.exit("분석할 프레임이 없어요. --start/--end 값을 확인해 주세요.")
    return np.array(rows, dtype=float), fps, file_fps, frames_for_video


# ----------------------------------------------------------------------------
# 3. 분석 구간 자동 선택
# ----------------------------------------------------------------------------
def auto_segment(data, still_px=2.0):
    """가장 긴 연속 검출 구간에서 ① 놓기 전 정지 구간 ② 가장 낮은 점(바닥) 이후를 잘라낸다."""
    ok = ~np.isnan(data[:, 2])
    best, cur_start, best_len = (0, 0), None, 0
    for k, v in enumerate(np.append(ok, False)):
        if v and cur_start is None:
            cur_start = k
        elif not v and cur_start is not None:
            if k - cur_start > best_len:
                best, best_len = (cur_start, k), k - cur_start
            cur_start = None
    s, e = best
    if e - s < 5:
        return None
    y = data[s:e, 3]
    low = int(np.argmax(y))                        # 화면에서 y가 가장 큰 곳 = 가장 낮은 곳
    e = s + low                                    # 바닥에 닿는 프레임은 충돌 중이라 제외
    y = data[s:e, 3]
    x = data[s:e, 2]
    moved = np.hypot(x - x[0], y - y[0]) > still_px
    if moved.any():
        first_move = int(np.argmax(moved))
        s = s + max(0, first_move - 1)             # 움직이기 직전 프레임부터
    return (s, e) if e - s >= 5 else None


# ----------------------------------------------------------------------------
# 4. 물리량 계산
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

    # 단위 질량당 역학적 에너지 (J/kg): ½v² + g·y  (y=0: 놓은 위치)
    e_mech = 0.5 * (vx ** 2 + vy ** 2) + G_THEORY * ys
    e_drift = float(np.percentile(e_mech, 95) - np.percentile(e_mech, 5))   # 튀는 값 몇 개에 휘둘리지 않게 5~95% 범위
    drop_energy = G_THEORY * float(np.ptp(ys)) or 1e-12

    return {
        "t": t, "x": x, "y": y, "y_fit": y_hat, "vx": vx, "vy": vy, "ay": ay, "e": e_mech,
        "g_fit": float(g_fit), "g_se": float(g_se), "r2": float(r2),
        "v0y": float(c1), "vx_fit": float(bx),
        "rmse_mm": float(np.sqrt(ss_res / len(t)) * 1000),
        "ay_mean": float(np.mean(ay)),
        "energy_drift_pct": e_drift / drop_energy * 100,
        "fit_coeffs": [float(c2), float(c1), float(c0)],
    }


# ----------------------------------------------------------------------------
# 5. 결과 저장
# ----------------------------------------------------------------------------
def save_csv(path, data, seg_range, ppm, res):
    s, e = seg_range
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_video_s", "x_px", "y_px", "radius_px", "used",
                    "t_s", "x_m", "y_m", "y_fit_m", "vx_mps", "vy_mps", "ay_mps2", "energy_Jpkg"])
        for k, row in enumerate(data):
            used = s <= k < e
            base = [int(row[0]), f"{row[1]:.5f}"] + ["" if np.isnan(v) else f"{v:.2f}" for v in row[2:5]] + [int(used)]
            if used:
                j = k - s
                base += [f"{res['t'][j]:.5f}", f"{res['x'][j]:.5f}", f"{res['y'][j]:.5f}", f"{res['y_fit'][j]:.5f}",
                         f"{res['vx'][j]:.4f}", f"{res['vy'][j]:.4f}", f"{res['ay'][j]:.3f}", f"{res['e'][j]:.4f}"]
            w.writerow(base)


def save_plot(path, res, title):
    t = res["t"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    ax = axs[0, 0]
    ax.plot(t, res["y"], "o", ms=3, label="measured")
    ax.plot(t, res["y_fit"], "-", lw=2, label=f"fit: g = {res['g_fit']:.3f} m/s²")
    ax.set(xlabel="t (s)", ylabel="y (m, up +)", title="Position")
    ax.legend(); ax.grid(alpha=.3)

    ax = axs[0, 1]
    ax.plot(t, res["vy"], "o", ms=3, label="vy (local fit)")
    ax.plot(t, res["v0y"] - res["g_fit"] * t, "-", lw=2, label="vy from fit")
    if abs(res["vx_fit"]) > 0.05:
        ax.plot(t, res["vx"], "s", ms=3, label="vx")
    ax.set(xlabel="t (s)", ylabel="v (m/s)", title="Velocity")
    ax.legend(); ax.grid(alpha=.3)

    ax = axs[1, 0]
    ax.plot(t, res["ay"], "o", ms=3, label="ay (local fit)")
    ax.axhline(-res["g_fit"], color="C1", lw=2, label=f"fit  {-res['g_fit']:.3f}")
    ax.axhline(-G_THEORY, color="k", ls="--", lw=1, label=f"theory {-G_THEORY:.3f}")
    ax.set(xlabel="t (s)", ylabel="a (m/s²)", title="Acceleration")
    lo = -G_THEORY * 2.2
    ax.set_ylim(lo, G_THEORY * 0.6)
    ax.legend(); ax.grid(alpha=.3)

    ax = axs[1, 1]
    ax.plot(t, res["e"], "o", ms=3)
    ax.set(xlabel="t (s)", ylabel="½v² + g·y  (J/kg)", title=f"Mechanical energy per mass (spread {res['energy_drift_pct']:.1f}%)")
    ax.grid(alpha=.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_annotated_video(path, frames, data, seg_range, fps):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), min(fps, 60), (w, h))
    s, e = seg_range
    trail = []
    for k, fr in enumerate(frames):
        img = fr.copy()
        x, y, r = data[k, 2:5]
        if not np.isnan(x):
            col = (0, 200, 0) if s <= k < e else (0, 200, 255)
            cv2.circle(img, (int(x), int(y)), int(max(r, 4)), col, 2)
            if s <= k < e:
                trail.append((int(x), int(y)))
        for p in trail:
            cv2.circle(img, p, 2, (255, 0, 0), -1)
        cv2.putText(img, f"t={data[k,1]:.3f}s", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        vw.write(img)
    vw.release()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="영상 속 공을 추적해 위치·속도·가속도·g를 계산합니다.")
    ap.add_argument("video")
    ap.add_argument("--pick", action="store_true", help="첫 프레임에서 공과 자 양 끝을 클릭해서 설정")
    ap.add_argument("--pick-time", type=float, default=0.0, help="--pick에 사용할 프레임 시각 (s)")
    ap.add_argument("--calib", help="설정 파일(json) 경로. --pick이면 여기에 저장")
    ap.add_argument("--color", choices=COLOR_PRESETS.keys(), help="공 색상 프리셋")
    ap.add_argument("--hsv", help="직접 HSV 범위: h1,s1,v1,h2,s2,v2")
    ap.add_argument("--scale-points", help="기준 물체 양 끝 픽셀: x1,y1,x2,y2")
    ap.add_argument("--scale-length", type=float, help="기준 물체 실제 길이 (m)")
    ap.add_argument("--px-per-m", type=float, help="1 m당 픽셀 수 (직접 지정)")
    ap.add_argument("--fps", type=float, help="실제 촬영 fps (슬로모션이면 꼭 지정)")
    ap.add_argument("--start", type=float, help="분석 시작 시각 (s, 영상 기준)")
    ap.add_argument("--end", type=float, help="분석 끝 시각 (s, 영상 기준)")
    ap.add_argument("--no-auto", action="store_true", help="정지·튕김 구간 자동 제외를 끔")
    ap.add_argument("--smooth", type=int, default=7, help="속도 계산에 쓰는 국소 2차 회귀 창 크기 (가속도는 2배+1)")
    ap.add_argument("--min-area", type=float, default=30, help="공으로 인정할 최소 면적 (px²)")
    ap.add_argument("--save-video", action="store_true", help="궤적을 그린 영상도 저장")
    ap.add_argument("--out", help="결과 폴더 (기본: 영상이름_result)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"영상을 열 수 없어요: {args.video}")
    fps0 = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.pick_time * fps0))
    ok, first = cap.read()
    cap.release()
    if not ok:
        sys.exit("첫 프레임을 읽을 수 없어요.")

    cfg = build_config(args, first)
    ppm = cfg["px_per_m"]
    data, fps, file_fps, frames = track(args.video, cfg, args)
    detected = int(np.sum(~np.isnan(data[:, 2])))
    print(f"프레임 {len(data)}개 중 {detected}개에서 공 검출  (분석 fps {fps:g}, 파일 fps {file_fps:g}, {ppm:.1f} px/m)")
    if detected < 5:
        sys.exit("공을 거의 찾지 못했어요. --pick으로 공 색을 다시 지정하거나 조명을 밝게 해 보세요.")

    if args.no_auto:
        ok = ~np.isnan(data[:, 2])
        idx = np.where(ok)[0]
        seg_range = (int(idx[0]), int(idx[-1]) + 1)
        data_seg = data[seg_range[0]:seg_range[1]]
        data_seg = data_seg[~np.isnan(data_seg[:, 2])]
    else:
        seg_range = auto_segment(data)
        if seg_range is None:
            sys.exit("연속으로 추적된 구간이 너무 짧아요. --no-auto 또는 --start/--end를 써 보세요.")
        data_seg = data[seg_range[0]:seg_range[1]]
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
    if args.save_video:
        save_annotated_video(os.path.join(out, "tracked.mp4"), frames, data, seg_range, fps)
    summary = {
        "video": args.video, "fps_used": fps, "px_per_m": ppm,
        "frames_total": len(data), "frames_detected": detected,
        "segment_frames": [int(seg_range[0]), int(seg_range[1])], "points_used": len(data_seg),
        "duration_s": float(res["t"][-1]),
        "g_fit": res["g_fit"], "g_standard_error": res["g_se"], "g_ci95": list(ci),
        "g_theory": G_THEORY, "relative_error_pct": err,
        "r2": res["r2"], "rmse_mm": res["rmse_mm"],
        "initial_vy": res["v0y"], "horizontal_v": res["vx_fit"],
        "mean_ay_from_differences": res["ay_mean"],
        "energy_spread_pct": res["energy_drift_pct"],
        "fit_y_t": {"c2": res["fit_coeffs"][0], "c1": res["fit_coeffs"][1], "c0": res["fit_coeffs"][2]},
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("========== 분석 결과 ==========")
    print(f"분석 구간      : 프레임 {seg_range[0]} ~ {seg_range[1]-1}  ({len(data_seg)}점, {res['t'][-1]:.3f} s)")
    print(f"중력가속도 g   : {res['g_fit']:.3f} ± {res['g_se']:.3f} m/s²  (95% 구간 {ci[0]:.3f} ~ {ci[1]:.3f})")
    print(f"이론값 대비    : {G_THEORY:.3f} m/s², 오차율 {err:.2f} %"
          + ("  → 이론값이 95% 구간 안" if ci[0] <= G_THEORY <= ci[1] else "  → 이론값이 95% 구간 밖 (계통 오차 의심)"))
    print(f"곡선 맞춤 품질 : R² = {res['r2']:.5f}, 잔차 RMS = {res['rmse_mm']:.2f} mm")
    print(f"초기 속도      : 수직 {res['v0y']:.3f} m/s, 수평 {res['vx_fit']:.3f} m/s")
    print(f"프레임별 가속도 평균: {res['ay_mean']:.3f} m/s² (국소 회귀, 참고용)")
    print(f"역학적 에너지  : 변동폭 {res['energy_drift_pct']:.1f} % (작을수록 에너지 보존에 가까움)")
    print(f"결과 폴더      : {out}/ (track.csv, analysis.png, summary.json{', tracked.mp4' if args.save_video else ''})")
    if fps == file_fps and file_fps <= 60 and err > 30:
        print("\n참고: 오차가 매우 크다면 슬로모션 영상일 수 있어요. 실제 촬영 fps를 --fps 240 처럼 지정해 보세요.")


if __name__ == "__main__":
    main()
