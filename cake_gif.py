#!/usr/bin/env python3
"""
Galaxy Cake Pixel Player
========================
把一段视频转成 GIF，再用「像素风 + 无缝循环」的方式播放（支持多格九宫格）。

用法（在脚本目录下执行）：
    python3 cake_gif.py                          # 用内置 VIDEO_SOURCE 转 GIF 并播放
    python3 cake_gif.py --video 生日.MP4          # 指定视频
    python3 cake_gif.py --gif cake.gif           # 直接播放已有 GIF（不转码）
    python3 cake_gif.py --grid 2x2               # 2×2 九宫格循环播放
    python3 cake_gif.py --pixel 4 --size 540     # 像素更大、窗口更小
    python3 cake_gif.py --export out.mp4         # 无界面：导出成 mp4（服务器/手机都能跑）
    python3 cake_gif.py --export out.gif         # 无界面：导出成 GIF

依赖：
    pip install opencv-python pillow numpy
    （想要窗口播放需要桌面环境；纯手机/服务器用 --export 即可）
"""
import argparse
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 可选：自动切换到同目录下的 .venv（找不到就正常继续） ----------
_VENV_DIR = os.path.join(BASE_DIR, ".venv")
_VENV_PY = os.path.join(_VENV_DIR, "bin", "python3")
if (
    os.path.exists(_VENV_PY)
    and os.path.realpath(sys.prefix) != os.path.realpath(_VENV_DIR)
    and not os.environ.get("CAKE_GIF_NO_VENV")
):
    os.execv(_VENV_PY, [_VENV_PY, os.path.abspath(__file__)] + sys.argv[1:])

import cv2
import numpy as np
from PIL import Image, ImageSequence

# ---------- 默认参数 ----------
VIDEO_SOURCE = os.path.join(BASE_DIR, "birthday.mp4")
GIF_PATH = os.path.join(BASE_DIR, "cake.gif")
GIF_SIZE = 720
GIF_MAX_FRAMES = 240

PIXEL_SIZE_FACTOR = 2
TARGET_SIZE = (720, 720)
LOOP_SECONDS = 0
FRAME_DELAY_MS = 30

GRID_ROWS, GRID_COLS = 1, 1
WINDOW_NAME = "Galaxy Cake Pixel Player [q/ESC quit]"


# ---------- 视频 → GIF ----------
def video_to_gif(video_path, gif_path, size=GIF_SIZE, max_frames=GIF_MAX_FRAMES):
    """把视频等间隔抽帧、居中裁成正方形、缩放后存成 GIF。返回帧数。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"打不开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, math.ceil(total / max_frames)) if total > 0 else 1

    frames = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            # 居中裁成正方形再缩放，避免被拉变形
            w, h = img.size
            side = min(w, h)
            img = img.crop(((w - side) // 2, (h - side) // 2,
                            (w + side) // 2, (h + side) // 2))
            img = img.resize((size, size), Image.LANCZOS)
            frames.append(img)
            if len(frames) >= max_frames:
                break
        index += 1
    cap.release()

    if not frames:
        raise RuntimeError("视频里读不到任何帧")

    duration_ms = max(20, int(round(1000 * step / fps)))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return len(frames)


# ---------- GIF 读取 ----------
def load_gif_frames(gif_path):
    frames = []
    with Image.open(gif_path) as im:
        for frame in ImageSequence.Iterator(im):
            frames.append(frame.convert("RGB").copy())
    if not frames:
        raise RuntimeError(f"GIF 里没有帧：{gif_path}")
    return frames


def gif_frame_duration(gif_path):
    with Image.open(gif_path) as im:
        return max(10, int(im.info.get("duration") or 33))


# ---------- 循环点检测：让首尾衔接更顺滑 ----------
def find_loop_point(frames):
    """在后半段找一帧，最接近第 0 帧、且差异远小于相邻帧的正常变化，就从这里循环。"""
    n = len(frames)
    if n < 12:
        return n
    small = np.stack(
        [np.asarray(f.convert("L").resize((96, 96)), dtype=np.float32) for f in frames]
    )
    base = float(np.median(
        [float(np.mean((small[i + 1] - small[i]) ** 2)) for i in range(n - 1)]
    ))
    if base <= 0:
        return n
    start = n // 2
    dists = [float(np.mean((small[k] - small[0]) ** 2)) for k in range(start, n)]
    best = start + int(np.argmin(dists))
    if dists[best - start] <= 4.0 * base:
        return best
    return n


# ---------- 像素化 ----------
def pixelate_frame(pil_image, pixel_size_factor=PIXEL_SIZE_FACTOR, target_size=TARGET_SIZE):
    """先缩小再最近邻放大，得到马赛克像素风；返回 BGR 数组。"""
    w, h = target_size
    factor = max(1, int(pixel_size_factor))
    img = pil_image.resize((w, h), Image.LANCZOS)
    if factor > 1:
        small = img.resize((max(1, w // factor), max(1, h // factor)), Image.BOX)
        img = small.resize((w, h), Image.NEAREST)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# ---------- 按 tick 组装一帧画面 ----------
def compose_frame(cell_frames, cell_ticks, tick, target_size, pixel_size_factor, grid):
    rows, cols = grid
    cell_w, cell_h = target_size
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for k, frames in enumerate(cell_frames):
        n = len(frames)
        if n == 0:
            continue
        tp = cell_ticks[k]
        pos = (tick % tp) * n / tp
        i = int(pos) % n
        alpha = pos - int(pos)
        if alpha > 1e-3 and n > 1:
            pil = Image.blend(frames[i], frames[(i + 1) % n], alpha)
        else:
            pil = frames[i]
        r, c = divmod(k, cols)
        canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = pixelate_frame(
            pil, pixel_size_factor, target_size
        )
    return canvas


# ---------- 窗口播放 ----------
def play_multi_gif_grid(gif_paths, grid=(GRID_ROWS, GRID_COLS), pixel_size_factor=PIXEL_SIZE_FACTOR,
                        target_size=TARGET_SIZE, loop_seconds=LOOP_SECONDS,
                        frame_delay_ms=FRAME_DELAY_MS, max_frames=None):
    all_frames = [load_gif_frames(p) for p in gif_paths]
    loop_points = [find_loop_point(f) for f in all_frames]
    cell_src = [f[:lp] for f, lp in zip(all_frames, loop_points)]
    rows, cols = grid
    n_cells = rows * cols
    cell_frames = [cell_src[i % len(cell_src)] for i in range(n_cells)]

    tick_s = frame_delay_ms / 1000.0
    cell_ticks = []
    for k in range(n_cells):
        src = k % len(cell_src)
        n_k = len(cell_src[src])
        if loop_seconds > 0:
            cell_ticks.append(max(1, int(round(loop_seconds / tick_s))))
        else:
            cell_ticks.append(max(1, int(round(n_k * gif_frame_duration(gif_paths[src]) / frame_delay_ms))))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    tick = 0
    print(f"播放中… 无缝循环（{loop_points[0]}/{len(all_frames[0])} 帧），"
          f"按 q 或 ESC 退出。")
    try:
        while True:
            canvas = compose_frame(cell_frames, cell_ticks, tick, target_size, pixel_size_factor, grid)
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
            tick += 1
    finally:
        cv2.destroyAllWindows()


# ---------- 无界面导出（手机 / 服务器可用） ----------
def export_video(gif_paths, out_path, grid=(GRID_ROWS, GRID_COLS), pixel_size_factor=PIXEL_SIZE_FACTOR,
                 target_size=TARGET_SIZE, loop_seconds=LOOP_SECONDS,
                 frame_delay_ms=FRAME_DELAY_MS, max_frames=None):
    all_frames = [load_gif_frames(p) for p in gif_paths]
    loop_points = [find_loop_point(f) for f in all_frames]
    cell_src = [f[:lp] for f, lp in zip(all_frames, loop_points)]
    rows, cols = grid
    n_cells = rows * cols
    cell_frames = [cell_src[i % len(cell_src)] for i in range(n_cells)]

    tick_s = frame_delay_ms / 1000.0
    cell_ticks = []
    for k in range(n_cells):
        src = k % len(cell_src)
        n_k = len(cell_src[src])
        if loop_seconds > 0:
            cell_ticks.append(max(1, int(round(loop_seconds / tick_s))))
        else:
            cell_ticks.append(max(1, int(round(n_k * gif_frame_duration(gif_paths[src]) / frame_delay_ms))))

    total_ticks = max(cell_ticks)
    if max_frames:
        total_ticks = min(total_ticks, max_frames)

    cell_w, cell_h = target_size
    width, height = cols * cell_w, rows * cell_h
    fps = max(1.0, 1000.0 / frame_delay_ms)
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out_path.lower().endswith(".mp4") else "MJPG"))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法写入：{out_path}")
    try:
        for tick in range(total_ticks):
            writer.write(compose_frame(cell_frames, cell_ticks, tick, target_size, pixel_size_factor, grid))
    finally:
        writer.release()
    print(f"已导出 {total_ticks} 帧 → {out_path}（{width}×{height} @ {fps:.1f}fps）")
    return total_ticks


def parse_grid(text):
    try:
        r, c = text.lower().split("x")
        return max(1, int(r)), max(1, int(c))
    except Exception:
        raise argparse.ArgumentTypeError("网格格式应为 2x2 这样的写法")


def main():
    ap = argparse.ArgumentParser(description="把视频转成 GIF 再像素风无缝循环播放")
    ap.add_argument("--video", default=VIDEO_SOURCE, help="源视频路径")
    ap.add_argument("--gif", default=GIF_PATH, help="GIF 路径（存在就直接用，不存在就转码）")
    ap.add_argument("-s", "--size", type=int, default=GIF_SIZE, help="每格边长，默认 720")
    ap.add_argument("--max-frames", type=int, default=GIF_MAX_FRAMES, help="GIF 最多帧数，默认 240")
    ap.add_argument("-p", "--pixel", type=int, default=PIXEL_SIZE_FACTOR, help="像素块大小，1=不像素化")
    ap.add_argument("-g", "--grid", type=parse_grid, default=(GRID_ROWS, GRID_COLS), help="网格，如 2x2")
    ap.add_argument("--loop-seconds", type=float, default=LOOP_SECONDS, help="一格循环几秒，0=按原始帧率")
    ap.add_argument("--delay", type=int, default=FRAME_DELAY_MS, help="每帧间隔毫秒，默认 30")
    ap.add_argument("--export", metavar="OUT", help="无界面导出到 mp4/gif")
    ap.add_argument("--reconvert", action="store_true", help="强制重新转码 GIF")
    args = ap.parse_args()

    target = (args.size, args.size)

    if args.reconvert or not os.path.exists(args.gif):
        if not os.path.exists(args.video):
            sys.exit(f"找不到视频：{args.video}\n先放一个视频进脚本目录，或用 --video 指定路径")
        n = video_to_gif(args.video, args.gif, args.size, args.max_frames)
        print(f"GIF 已生成：{args.gif}（{n} 帧）")

    if args.export:
        if args.export.lower().endswith(".gif"):
            frames = load_gif_frames(args.gif)
            lp = find_loop_point(frames)
            frames = frames[:lp]
            if args.max_frames:
                frames = frames[: args.max_frames]
            frames[0].save(args.export, save_all=True, append_images=frames[1:],
                           duration=args.delay, loop=0, optimize=True)
            print(f"已导出 GIF：{args.export}（{len(frames)} 帧）")
        else:
            export_video([args.gif], args.export, grid=args.grid, pixel_size_factor=args.pixel,
                         target_size=target, loop_seconds=args.loop_seconds,
                         frame_delay_ms=args.delay, max_frames=args.max_frames)
        return

    play_multi_gif_grid([args.gif], grid=args.grid, pixel_size_factor=args.pixel,
                        target_size=target, loop_seconds=args.loop_seconds,
                        frame_delay_ms=args.delay)


if __name__ == "__main__":
    main()
