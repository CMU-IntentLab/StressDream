"""Visualization utilities for Ctrl-World steering outputs."""

import logging

import imageio
import numpy as np

VIEW_NAMES = ["left", "right", "wrist"]


def _load_font(size=9):
    from PIL import ImageFont
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def overlay_rewards_on_frame(frame, frame_idx, total_frames, instruction, reward_curves):
    """Overlay reward curves onto a single video frame.

    Args:
        frame: (H, W, 3) uint8 numpy array.
        frame_idx: int, current frame index.
        total_frames: int, total number of frames in the clip.
        instruction: str, task instruction text displayed at the top.
        reward_curves: dict mapping metric name -> list of float values, one per frame.

    Returns:
        (margin_height + H, W, 3) uint8 numpy array with overlaid text and sparklines.
    """
    from PIL import Image, ImageDraw

    LINE_H = 13
    num_metrics = sum(1 for c in reward_curves.values() if c)
    margin_height = 2 * LINE_H + num_metrics * LINE_H + 6
    H, W, C = frame.shape
    full = np.zeros((margin_height + H, W, C), dtype=np.uint8)
    full[:] = (40, 40, 40)
    full[margin_height:] = frame
    img = Image.fromarray(full)
    draw = ImageDraw.Draw(img)
    font = _load_font(9)
    y = 3
    short = instruction[:50] + ".." if len(instruction) > 50 else instruction
    draw.text((6, y), f"Task: {short}", fill=(200, 200, 200), font=font)
    y += 13
    draw.text((6, y), f"Frame {frame_idx}/{total_frames - 1}", fill=(180, 180, 180), font=font)
    y += 13
    bar_w = min(W // 4, 100)
    spark_w = min(W // 3, 140)
    for name, curve in reward_curves.items():
        if not curve:
            continue
        val = curve[min(frame_idx, len(curve) - 1)]
        g = int(min(255, 80 + 350 * max(0, val)))
        color = (80, g, 80)
        draw.text((6, y), f"{name}: {val:.3f}", fill=color, font=font)
        bx = 130
        draw.rectangle([bx, y + 1, bx + bar_w, y + 9], fill=(80, 80, 80))
        fw = int(bar_w * max(0, min(1, val)))
        if fw > 0:
            draw.rectangle([bx, y + 1, bx + fw, y + 9], fill=color)
        sx = bx + bar_w + 8
        sh = 10
        if frame_idx > 0 and spark_w > 10:
            draw.rectangle([sx, y, sx + spark_w, y + sh], fill=(60, 60, 60))
            pts = []
            for i in range(min(frame_idx + 1, len(curve))):
                px = sx + int(i / max(total_frames - 1, 1) * spark_w)
                py = y + sh - int(max(0, min(1, curve[i])) * sh)
                py = max(y, min(y + sh, py))
                pts.append((px, py))
            if len(pts) >= 2:
                draw.line(pts, fill=(100, 255, 100), width=1)
            cx = sx + int(frame_idx / max(total_frames - 1, 1) * spark_w)
            cy = y + sh - int(max(0, min(1, val)) * sh)
            cy = max(y, min(y + sh, cy))
            draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 255, 100))
        y += 13
    return np.array(img)


def save_overlay_video(video_frames, reward_curves_per_view, instruction, output_path, fps=4):
    """Render a 3-view overlay video with reward sparklines burned in.

    Args:
        video_frames: list of 3 arrays, each (T, H, W, 3) uint8.
        reward_curves_per_view: list of 3 dicts, each {name: [float]*T}.
        instruction: str, task instruction text.
        output_path: str, destination .mp4 path.
        fps: int, output frame rate.
    """
    T = video_frames[0].shape[0]
    out_frames = []
    for t in range(T):
        panels = []
        for vi in range(3):
            panel = overlay_rewards_on_frame(
                video_frames[vi][t],
                t,
                T,
                f"[{VIEW_NAMES[vi]}] {instruction}",
                reward_curves_per_view[vi],
            )
            panels.append(panel)
        out_frames.append(np.concatenate(panels, axis=1))
    imageio.mimwrite(output_path, out_frames, fps=fps, codec="libx264")
    logging.info("Overlay video saved to %s", output_path)


def pixels_to_uint8_views(pixels_cpu, num_views=3):
    """Convert raw decoded pixels to per-view uint8 frame arrays.

    Args:
        pixels_cpu: numpy array of shape (num_views, T, C, H, W) in [-1, 1],
                    or a torch tensor that has already been moved to CPU.
        num_views: int, number of camera views (default 3).

    Returns:
        List of num_views arrays, each shaped (T, H, W, 3) uint8.
    """
    import torch

    if isinstance(pixels_cpu, torch.Tensor):
        pixels_cpu = pixels_cpu.float().numpy()

    views = []
    for v in range(num_views):
        # pixels_cpu[v]: (T, C, H, W) in [-1, 1]
        frames = ((pixels_cpu[v] / 2.0 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        # (T, C, H, W) -> (T, H, W, C)
        frames = frames.transpose(0, 2, 3, 1)
        views.append(frames)
    return views
